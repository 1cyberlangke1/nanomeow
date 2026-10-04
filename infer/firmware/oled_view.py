# -*- coding: utf-8 -*-
"""OLED 可视化模拟：让上板固件在 Cortex-M3 模拟器里真跑，把软件 I2C 的位流还原成 SSD1306 屏幕。

输入：`build_firmware.py` 真交叉编译出的固件 ELF、要喂给 USART1 的一行字节。
输出：128x64 屏幕的字符画（默认半块字符 ▀ ▄ █，一个字符 = 上下两个像素），外加串口回显。
预期行为：跑的是真机指令，不是行为模型。板级外设只多打一个桩 —— 拦 GPIOB 的读写，按 I2C 时序
          （START / STOP、SCL 上升沿采样、第 9 个时钟的 ACK）把位流还原成字节，再按 SSD1306
          的命令集重建显示 RAM。屏幕上显示什么，这里就打什么。

本文件只认协议，不认固件的画法：不假设「哪一屏是答案」「刷新怎么写」「有没有清屏」，
所以改 C 源码（换布局、去前缀、改刷新策略）都不用动模拟器。屏幕内容每变一次就记一帧，
CLI 把这几帧打出来。

为什么要给 ACK 打桩：固件在 nm_i2c_put 里读 IDR 判断从机有没有应答，读不到低电平就会把后面
所有字节都掐掉（`ok` 门）；所以第 9 个时钟必须由模拟出来的 SSD1306 把 SDA 拉低。

用法：
    python infer/firmware/oled_view.py --text 你好
    python infer/firmware/oled_view.py --text 你好 --all-frames
    python infer/firmware/oled_view.py --text 你好 --frame 12
    python infer/firmware/oled_view.py --text 你好 --plain        # 用 '#' / '.'，兼容性最好
    python infer/firmware/oled_view.py --scl-pin 6 --sda-pin 7    # 换接线时不用改本文件
"""

import argparse
import pathlib
import sys

import run_m3

# 与 infer/c/platform/nm_stm32f103.h 一致的 GPIOB 寄存器（软件 I2C 只用 BSRR / BRR / IDR）。
GPIOB_BSRR = 0x40010C10
GPIOB_BRR = 0x40010C14
GPIOB_IDR = 0x40010C08

# 与 infer/c/display/nm_oled.h 一致的屏参数与 SSD1306 的 I2C 帧格式（都是器件规格，不是固件画法）。
OLED_W, OLED_H = 128, 64
OLED_PAGES = OLED_H // 8
CTRL_COMMAND = 0x00      # 控制字节 0x00 = 后面是命令流
CTRL_DATA = 0x40         # 控制字节 0x40 = 后面是显示数据流


class I2cDecoder:
    """软件 I2C 的位级解码器：喂引脚电平变化，吐字节。

    输入：每次引脚电平变化调一次 update(scl, sda)。
    输出：每收满 8 位回调一次 on_byte；START / STOP 时回调 on_start / on_stop。
    预期行为：SCL 高时 SDA 由高变低是 START、由低变高是 STOP；数据在 SCL 上升沿采样、MSB 在前；
              第 9 个时钟是 ACK 位 —— 本模型里的从机永远应答，所以只把这段时间标成 ack_phase
              （上层据此在读 IDR 时把 SDA 拉低），不采样它的值。
    """

    def __init__(self, on_byte, on_start=None, on_stop=None):
        self.scl = 0
        self.sda = 0
        self.active = False          # START 之后、STOP 之前
        self.bits = 0
        self.nbits = 0
        self.awaiting_ack = False
        self.ack_phase = False       # 第 9 个时钟内为 True（从机该把 SDA 拉低）
        self._on_byte = on_byte
        self._on_start = on_start
        self._on_stop = on_stop

    def update(self, scl, sda):
        """输入：当前 SCL / SDA 电平（0 或 1）；输出：无。预期行为：见类注释。"""
        prev_scl, prev_sda = self.scl, self.sda
        self.scl, self.sda = scl, sda
        if scl == prev_scl and sda == prev_sda:
            return
        if scl:                                  # SCL 高时 SDA 的跳变就是起止条件
            if prev_sda and not sda:
                self._begin()
                return
            if not prev_sda and sda:
                self._end()
                return
        if not self.active:
            return
        if not prev_scl and scl:                 # 上升沿采样
            if self.awaiting_ack:
                self.ack_phase = True
                self.awaiting_ack = False
            else:
                self.bits = (self.bits << 1) | sda
                self.nbits += 1
                if self.nbits == 8:
                    byte, self.bits, self.nbits = self.bits, 0, 0
                    self.awaiting_ack = True
                    self._on_byte(byte)
        elif prev_scl and not scl:
            self.ack_phase = False

    def _begin(self):
        self.active = True
        self.bits = self.nbits = 0
        self.awaiting_ack = self.ack_phase = False
        if self._on_start is not None:
            self._on_start()

    def _end(self):
        self.active = False
        self.bits = self.nbits = 0
        self.awaiting_ack = self.ack_phase = False
        if self._on_stop is not None:
            self._on_stop()


class OledScreen:
    """SSD1306 的显示 RAM 模型：只实现本固件用到的命令，其余按数据手册的参数个数跳过。

    输入：command(b) 喂命令字节流、data(b) 喂显示数据字节流（对应 I2C 控制字节 0x00 / 0x40）。
    输出：ram 是 8 页 x 128 列的显示 RAM，字节的 bit0..bit7 对应那一页的第 0..7 个像素行。
    预期行为：0x21 / 0x22 设列 / 页窗口，0x20 选寻址模式，数据按模式自动推进地址；
              参数个数表覆盖常见命令，没见过的命令当无参数处理，不会把参数吃成命令。
    """

    # 命令的参数个数。0x00..0x1F（列起始）、0x40..0x7F（起始行）、0xB0..0xB7（页起始）都是 0 参数。
    _NARGS = {
        0x20: 1, 0x21: 2, 0x22: 2, 0x2E: 0, 0x2F: 0,
        0x81: 1, 0x8D: 1, 0xA4: 0, 0xA5: 0, 0xA6: 0, 0xA7: 0,
        0xA8: 1, 0xAD: 1, 0xAE: 0, 0xAF: 0,
        0xB1: 1, 0xB2: 1, 0xB3: 1,
        0xD3: 1, 0xD5: 1, 0xD9: 1, 0xDA: 1, 0xDB: 1,
    }

    def __init__(self, width=OLED_W, height=OLED_H):
        self.width, self.height = width, height
        self.pages = height // 8
        self.ram = bytearray(self.pages * width)
        self.addr_mode = 2               # 上电默认页寻址
        self.col, self.page = 0, 0
        self.col_lo, self.col_hi = 0, width - 1
        self.page_lo, self.page_hi = 0, self.pages - 1
        self._cmd, self._args, self._need = None, [], 0

    @staticmethod
    def _nargs(b):
        """输入：命令字节；输出：它后面跟几个参数字节。预期行为：见 _NARGS 注释。"""
        if b <= 0x1F or 0x40 <= b <= 0x7F or 0xB0 <= b <= 0xB7:
            return 0
        return OledScreen._NARGS.get(b, 0)

    def command(self, b):
        """输入：命令流里的一个字节；输出：无。预期行为：收齐参数再执行，参数不会被当成命令。"""
        if self._cmd is not None:
            self._args.append(b)
            if len(self._args) >= self._need:
                self._exec(self._cmd, self._args)
                self._cmd, self._args, self._need = None, [], 0
            return
        n = self._nargs(b)
        if n:
            self._cmd, self._args, self._need = b, [], n
        else:
            self._exec(b, [])

    def _exec(self, cmd, args):
        """输入：一条命令与它的参数；输出：无。预期行为：只改本固件用到的那几条，其余忽略。"""
        if cmd == 0x21:
            self.col_lo, self.col_hi = args[0], args[1]
            self.col = self.col_lo
        elif cmd == 0x22:
            self.page_lo, self.page_hi = args[0], args[1]
            self.page = self.page_lo
        elif cmd == 0x20:
            self.addr_mode = args[0] & 0x03
        elif cmd <= 0x0F:
            self.col = (self.col & 0xF0) | cmd
        elif cmd <= 0x1F:
            self.col = (self.col & 0x0F) | ((cmd & 0x0F) << 4)
        elif 0xB0 <= cmd <= 0xB7:
            self.page = cmd & 0x07
        # 其余（起始行、对比度、电荷泵、VCOMH…）不改显示 RAM，忽略

    def data(self, b):
        """输入：显示数据流里的一个字节；输出：无。预期行为：写进当前地址再按寻址模式推进。"""
        if 0 <= self.page < self.pages and 0 <= self.col < self.width:
            self.ram[self.page * self.width + self.col] = b & 0xFF
        self._advance()

    def _advance(self):
        """输入：无；输出：无。预期行为：按寻址模式推进列 / 页，越界回窗口起点（数据手册的自动换行）。"""
        if self.addr_mode == 0:                  # 水平寻址：先横后竖
            self.col += 1
            if self.col > self.col_hi:
                self.col = self.col_lo
                self.page += 1
                if self.page > self.page_hi:
                    self.page = self.page_lo
        elif self.addr_mode == 1:                # 垂直寻址：先竖后横
            self.page += 1
            if self.page > self.page_hi:
                self.page = self.page_lo
                self.col += 1
                if self.col > self.col_hi:
                    self.col = self.col_lo
        else:                                    # 页寻址：只横着走，走到行尾回列首
            self.col += 1
            if self.col >= self.width:
                self.col = 0


class OledBus:
    """GPIOB + 软件 I2C + SSD1306 的合成模型：固件写寄存器，这里出屏幕内容。

    输入：write(reg, value) 报告固件对 GPIOB 的每一次写；read_idr() 报告固件读 IDR 时看到的电平。
    输出：screen 是重建出来的显示 RAM；frames 是屏幕内容每一次变化后的快照（相邻重复的并成一帧）。
    预期行为：BSRR 置位 / BRR 清位更新 ODR 影子寄存器，ODR 上两个引脚的电平变化喂给 I2C 解码器；
              解码出的字节按「第 1 个 = 器件地址（不校验）、第 2 个 = 控制字节、其余 = 载荷」分派。
              这里不认识任何固件的画法 —— 只认 I2C 与 SSD1306 的协议。
    """

    def __init__(self, scl_pin=8, sda_pin=9):
        self.scl_bit = 1 << scl_pin
        self.sda_bit = 1 << sda_pin
        self.odr = 0
        self.screen = OledScreen()
        self.transactions = 0
        self.frames = []
        self._addr = None
        self._ctrl = None
        self.decoder = I2cDecoder(self._byte, self._start, self._stop)

    def write(self, reg, value):
        """输入：GPIOB 寄存器地址与写入值；输出：无。预期行为：只跟踪影响位流的 BSRR / BRR。"""
        if reg == GPIOB_BSRR:
            self.odr = ((self.odr | (value & 0xFFFF)) & ~((value >> 16) & 0xFFFF)) & 0xFFFF
        elif reg == GPIOB_BRR:
            self.odr = (self.odr & ~(value & 0xFFFF)) & 0xFFFF
        else:
            return                               # CRL / CRH / IDR 写：不影响位流
        self.decoder.update(1 if self.odr & self.scl_bit else 0,
                            1 if self.odr & self.sda_bit else 0)

    def read_idr(self):
        """输入：无；输出：固件读 IDR 时应看到的 16 位电平。
        预期行为：平时就是 ODR 的回读；在 ACK 时钟里由「从机」把 SDA 拉低 —— 不给这个低电平，
                  固件的 ok 门会把后面所有字节都掐掉。"""
        if self.decoder.ack_phase:
            return self.odr & ~self.sda_bit
        return self.odr

    def _start(self):
        self._addr = self._ctrl = None

    def _byte(self, b):
        """输入：解码出来的一个字节；输出：无。预期行为：地址 → 控制字节 → 载荷，三段式分派。"""
        if self._addr is None:
            self._addr = b                       # 器件地址（0x78），本模型不校验
        elif self._ctrl is None:
            self._ctrl = b                       # 控制字节：0x00 命令流 / 0x40 数据流
        elif self._ctrl == CTRL_COMMAND:
            self.screen.command(b)
        else:
            self.screen.data(b)

    def _stop(self):
        """输入：无；输出：无。预期行为：事务收尾 —— 数一次事务；屏幕内容变了就记一帧。"""
        self.transactions += 1
        snap = bytes(self.screen.ram)
        if not self.frames or self.frames[-1] != snap:
            self.frames.append(snap)


def run_with_oled(elf_path, uart_in, scl_pin=8, sda_pin=9, max_steps=run_m3.MAX_STEPS, dwt=True):
    """输入：ELF 路径、喂给 USART1 的字节、OLED 的 SCL / SDA 引脚号、是否打开 DWT 周期桩。
    输出：(固件吐回串口的字节, OledBus)。

    预期行为：完全复用 run_m3 的模拟器（同一套 RCC / USART1 打桩、同一套停机判据），只多挂一个
              GPIOB 钩子；屏幕内容就是固件真发出去的 I2C 字节重建出来的。引脚号默认 PB8 / PB9，
              与 keil_demo/User/config.h 一致，换线时用命令行参数覆盖，不用改本文件。
              dwt 默认开：屏上右下角那行 tps 要读到非零的 DWT_CYCCNT 才画得出来，模拟器里没有
              硬件时钟，就由 run_m3 用指令数当周期数打桩（值只用于看显示位置与量级）。
    """
    bus = OledBus(scl_pin=scl_pin, sda_pin=sda_pin)
    out, _ = run_m3.run_firmware(elf_path, uart_in, max_steps=max_steps, gpio=bus, dwt=dwt)
    return out, bus


def count_lit(ram):
    """输入：显示 RAM；输出：点亮的像素个数。预期行为：逐字节数 1 的个数，给摘要和闸门用。"""
    return sum(bin(b).count("1") for b in ram)


def render_screen(ram, width=OLED_W, height=OLED_H, plain=False):
    """输入：页组织的显示 RAM、屏尺寸、是否用纯 ASCII 字符。
    输出：屏幕的字符画（多行，行尾无换行）。

    预期行为：默认一个字符画上下两个像素（▀ ▄ █），128x64 打成 128 列 x 32 行，长宽比接近真屏；
              plain=True 时一个像素一个字符（'#' 亮 / '.' 灭），128 列 x 64 行，任何终端都不会错位。
    """
    need = (height // 8) * width
    if len(ram) < need:
        raise ValueError("显示 RAM 只有 %d 字节，至少要 %d" % (len(ram), need))

    def lit(x, y):
        return (ram[(y // 8) * width + x] >> (y % 8)) & 1

    if plain:
        return "\n".join("".join("#" if lit(x, y) else "." for x in range(width))
                          for y in range(height))
    rows = []
    for y in range(0, height, 2):
        rows.append("".join((" ", "▀", "▄", "█")[lit(x, y) | (lit(x, y + 1) << 1)]
                            for x in range(width)))
    return "\n".join(rows)


def main():
    """输入：命令行；输出：屏幕字符画 + 串口回显。预期行为：ELF 不存在就报错提示先构建。"""
    ap = argparse.ArgumentParser(description="把上板固件的 OLED 屏幕画在终端里（真跑 Cortex-M3 模拟）")
    ap.add_argument("--elf", default=str(run_m3.ROOT / "build" / "firmware" / "nanomeow.elf"))
    ap.add_argument("--text", default="你好", help="喂给 USART1 的一行内容（会自动补换行）")
    ap.add_argument("--scl-pin", type=int, default=8, help="OLED SCL 引脚（默认 PB8，见 keil_demo/User/config.h）")
    ap.add_argument("--sda-pin", type=int, default=9, help="OLED SDA 引脚（默认 PB9）")
    ap.add_argument("--frame", type=int, default=None, help="只打第 N 帧（序号见 --all-frames 的表头）")
    ap.add_argument("--all-frames", action="store_true", help="把每一帧都打出来（回放显示过程）")
    ap.add_argument("--plain", action="store_true", help="用 '#' / '.' 画，不用半块字符")
    ap.add_argument("--build", action="store_true", help="先跑 build_firmware.py 再模拟")
    ap.add_argument("--no-dwt", dest="dwt", action="store_false",
                    help="关掉 DWT 周期桩（关掉后屏上不画 tps，因为固件读到的周期恒为 0）")
    args = ap.parse_args()

    if args.build:
        import subprocess
        subprocess.run([sys.executable,
                        str(pathlib.Path(__file__).resolve().parent / "build_firmware.py")], check=True)
    elf = pathlib.Path(args.elf)
    if not elf.exists():
        raise SystemExit("找不到 %s，先跑 python infer/firmware/build_firmware.py" % elf)

    out, bus = run_with_oled(elf, args.text.encode("utf-8") + b"\n",
                             scl_pin=args.scl_pin, sda_pin=args.sda_pin, dwt=args.dwt)

    write = sys.stdout.buffer.write

    def show(idx, why):
        """输入：帧号与说明；输出：无。预期行为：把那一帧连同表头打出来。"""
        write(("\n--- 第 %d/%d 帧，%s（点亮 %d 像素）---\n"
               % (idx, len(bus.frames) - 1, why, count_lit(bus.frames[idx]))).encode("utf-8"))
        write((render_screen(bus.frames[idx], plain=args.plain) + "\n").encode("utf-8"))

    write(("串口回显：%s\n" % out.decode("utf-8", "replace")).encode("utf-8"))
    write(("屏幕内容一共变了 %d 次（I2C 事务 %d 次）。\n"
           % (len(bus.frames), bus.transactions)).encode("utf-8"))

    if args.all_frames:
        for i in range(len(bus.frames)):
            show(i, "回放")
    elif args.frame is not None:
        if not 0 <= args.frame < len(bus.frames):
            raise SystemExit("帧号要在 0..%d 之间" % (len(bus.frames) - 1))
        show(args.frame, "指定")
    else:
        fullest = max(range(len(bus.frames)), key=lambda i: count_lit(bus.frames[i]))
        show(fullest, "内容最多的一帧")
        if fullest != len(bus.frames) - 1:
            show(len(bus.frames) - 1, "最后一帧（跑完停在屏上的）")
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
