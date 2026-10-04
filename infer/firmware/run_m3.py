# -*- coding: utf-8 -*-
"""在 Cortex-M3 模拟器（Unicorn）上**真正执行上板固件**，给「编译通过」补上「跑起来」的证据。

输入：`build_firmware.py` 产出的 ELF（真交叉编译 + 真链接的那个镜像），以及要喂给 USART1 的一行字节。
输出：固件从 USART1 吐回来的字节（stdout），可选 `--count` 给出精确执行的指令条数。
预期行为：按 ELF 的 PT_LOAD 段把镜像铺进 Flash/RAM，从复位向量取 SP/PC 起跑；板级外设只打两个桩 ——
          RCC 的 ready 位（HSERDY/PLLRDY/SWS）和 USART1 的 SR/DR，其余内存照实读写。
          跑的是**真机指令**，不是行为模型；指令条数是精确计数，不是估算。
          需要 `pip install unicorn`（纯 Python 包，只进本项目 .venv，不碰系统）。

停机判据：固件生成完会回到 `nm_uart_getc` 的阻塞轮询（RXNE 恒 0），此时它对 USART1_SR 会连续空读
          成千上万次 —— 数这个连续空读次数，超过阈值就认定这一轮结束，比「猜一个静默窗口」可靠。

用法：
    python infer/firmware/run_m3.py --text 你好
    python infer/firmware/run_m3.py --text 你好 --count
    python infer/firmware/run_m3.py --build --text 你好        # 先跑 build_firmware.py 再模拟
"""

import argparse
import pathlib
import struct
import sys

FW_DIR = pathlib.Path(__file__).resolve().parent
ROOT = FW_DIR.parents[1]

FLASH_BASE, FLASH_SIZE = 0x08000000, 64 * 1024
RAM_BASE, RAM_SIZE = 0x20000000, 20 * 1024
APB_BASE, APB_SIZE = 0x40000000, 0x30000      # APB1 + APB2 外设窗口
SCS_BASE, SCS_SIZE = 0xE0000000, 0x10000      # 内核私有区（本固件不开中断，但保留映射）

RCC_CR, RCC_CFGR = 0x40021000, 0x40021004
USART1_SR, USART1_DR = 0x40013800, 0x40013804

DWT_CYCCNT = 0xE0001004   # 内核私有区里的 32 位周期计数器（固件用它量 tps）

# GPIOB：OLED 的软件 I2C 走这一组寄存器。默认不挂，只有传了 gpio 钩子（oled_view.py）时才拦。
GPIOB_BASE = 0x40010C00
GPIOB_IDR = GPIOB_BASE + 0x08
GPIOB_END = GPIOB_BASE + 0x18

SR_TXE, SR_TC, SR_RXNE = 0x80, 0x40, 0x20     # 参考手册：TXE/TC 恒为 1（发送永不阻塞）
IDLE_SR = 50000                               # 连续空读 SR 超过这个次数 ⇒ 固件在等输入，本轮结束
DWT_CHUNK = 65536                             # 打开 DWT 桩时每块跑多少条指令（块间更新周期计数）
MAX_STEPS = 400_000_000                       # 保险丝：撞上就报错，不假装跑完

PT_LOAD = 1


class _Idle(Exception):
    """内部哨兵：固件回到「等串口输入」的阻塞轮询时，用它把模拟停下来。"""


def load_elf(path):
    """输入：ELF 路径；输出：[(物理地址, 该段在文件里的字节)]。

    预期行为：只读 PT_LOAD 段，自己解析程序头，不依赖 objdump（x86 的 objdump 读 ARM 不可靠）。
    """
    data = pathlib.Path(path).read_bytes()
    if data[:4] != b"\x7fELF":
        raise ValueError("%s 不是 ELF 文件" % path)
    if data[4] != 1:
        raise ValueError("%s 不是 32 位 ELF（本工具只处理 armv7m-none-eabi）" % path)
    endian = "<" if data[5] == 1 else ">"
    phoff, = struct.unpack_from(endian + "I", data, 0x1C)
    phentsize, phnum = struct.unpack_from(endian + "HH", data, 0x2A)
    segs = []
    for i in range(phnum):
        p_type, p_offset, _vaddr, p_paddr, p_filesz, _memsz, _flags, _align = \
            struct.unpack_from(endian + "IIIIIIII", data, phoff + i * phentsize)
        if p_type == PT_LOAD and p_filesz:
            segs.append((p_paddr, data[p_offset:p_offset + p_filesz]))
    if not segs:
        raise ValueError("%s 里没有可加载段" % path)
    return segs


def run_firmware(elf_path, uart_in, max_steps=MAX_STEPS, count=False, idle_sr=IDLE_SR, gpio=None,
                 dwt=False, dwt_chunk=DWT_CHUNK):
    """输入：ELF 路径、要喂给 USART1 的字节、指令上限、是否精确计数、空读阈值、GPIOB 钩子、DWT 桩。
    输出：(固件吐出的字节, 已执行指令条数或 None)。

    预期行为：从复位向量起跑，跑到固件阻塞等输入为止；跑到一半崩了就报错，不吞。
              `gpio` 不是 None 时，额外拦 GPIOB 的读写并转给它（`write(寄存器地址, 值)` /
              `read_idr()`）—— 这是给 oled_view.py 还原 OLED 位流用的，不传时行为与从前完全一样。
              `dwt` 为真时额外打一个桩：把「已执行的指令数」当周期数写进 DWT_CYCCNT，固件读它就能
              算出非零的 tps。板子上 CYCCNT 是硬件按 72 MHz 自增的，模拟器里没有这个时钟，所以用
              指令数代替（Cortex-M3 上大多数指令 1 个周期）—— 屏上那个 tps 是**模拟值**，只用来
              看显示位置与量级，真机走真 DWT。不传时行为与从前完全一样。
    """
    import faulthandler

    from unicorn import (Uc, UcError, UC_ARCH_ARM, UC_MODE_THUMB, UC_MODE_MCLASS,
                         UC_HOOK_BLOCK, UC_HOOK_MEM_READ, UC_HOOK_MEM_WRITE)
    from unicorn.arm_const import UC_ARM_REG_SP, UC_ARM_REG_PC

    # Unicorn 在 Windows 上把 TCG 缓冲按 MEM_RESERVE 预留、靠自己的 VEH 在首次触碰时补 MEM_COMMIT，
    # 于是每次 uc_mem_map 都会抛一个**被它自己吃掉**的 access violation（上游 issue #2264，修法是
    # AddVectoredExceptionHandler(1, ...)，2.1.4 还没有）。faulthandler 的 VEH 优先级更高，会抢先把它
    # 打成「Windows fatal exception」，看着像模拟崩了其实没有。所以模拟期间先关掉它，跑完原样恢复。
    was_faulthandler = faulthandler.is_enabled()
    if was_faulthandler:
        faulthandler.disable()

    mu = Uc(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS)
    for base, size in ((FLASH_BASE, FLASH_SIZE), (RAM_BASE, RAM_SIZE),
                       (APB_BASE, APB_SIZE), (SCS_BASE, SCS_SIZE)):
        mu.mem_map(base, size)
    for paddr, blob in load_elf(elf_path):
        mu.mem_write(paddr, blob)

    pending = bytearray(uart_in)              # 还没被固件取走的输入
    out = bytearray()
    steps = [0]
    idle = [0]                                # 连续空读 SR 的次数
    rcc = {RCC_CR: 0, RCC_CFGR: 0}            # RCC 影子寄存器：板子上的 ready 位是硬件置的
    cycles = [0]                              # DWT 桩：已执行指令数，当周期数用

    def on_write(uc, access, addr, size, value, ud):
        """输入：一次内存写；输出：无。

        预期行为：Unicorn 的写钩子在**写入之前**触发，所以这里只记影子值（用 value 参数），
                  真正「读回来」的补位交给 on_read —— 直接在这里改内存会被紧随其后的写入覆盖。
        """
        if addr in rcc:
            rcc[addr] = value & 0xFFFFFFFF
        elif addr == USART1_DR:
            out.append(value & 0xFF)
            idle[0] = 0                       # 又吐了一个字节 ⇒ 还在生成，空读计数清零
        elif gpio is not None and GPIOB_BASE <= addr < GPIOB_END:
            gpio.write(addr, value & 0xFFFFFFFF)

    def on_read(uc, access, addr, size, value, ud):
        """输入：一次内存读；输出：无。预期行为：读钩子先于访存触发，把打桩值铺进内存即可。"""
        if addr == RCC_CR:                     # HSEON/PLLON 一置位，HSERDY/PLLRDY 就跟着有效
            v = rcc[RCC_CR]
            if v & (1 << 16):
                v |= 1 << 17
            if v & (1 << 24):
                v |= 1 << 25
            uc.mem_write(addr, struct.pack("<I", v))
        elif addr == RCC_CFGR:                 # SW 选 PLL 时，SWS 立刻跟上
            v = rcc[RCC_CFGR]
            if (v & 0x3) == 0x2:
                v = (v & ~0xC) | 0x8
            uc.mem_write(addr, struct.pack("<I", v))
        elif addr == USART1_SR:
            uc.mem_write(addr, struct.pack("<I",
                                           SR_TXE | SR_TC | (SR_RXNE if pending else 0)))
            if not pending:
                idle[0] += 1
                if out and idle[0] > idle_sr:
                    raise _Idle()
        elif addr == USART1_DR:
            uc.mem_write(addr, struct.pack("<I", pending.pop(0) if pending else 0))
        elif addr == GPIOB_IDR and gpio is not None:
            uc.mem_write(addr, struct.pack("<I", gpio.read_idr() & 0xFFFFFFFF))
        elif addr == DWT_CYCCNT and dwt:
            # 读钩子先于访存：把当前累计的指令数铺进去，固件读 CYCCNT 就拿到它。
            uc.mem_write(addr, struct.pack("<I", cycles[0] & 0xFFFFFFFF))

    def thumb_len(blob, off):
        """输入：块字节与偏移；输出：这条 Thumb 指令占几个字节（2 或 4）。

        预期行为：按 ARM 的长度判定规则 —— 半字高 5 位是 0b11101/11110/11111 才是 32 位指令，
                  其余都是 16 位。这是数指令条数的唯一需要知道的解码规则。
        """
        hw = blob[off] | (blob[off + 1] << 8)
        return 4 if (hw >> 11) in (0x1D, 0x1E, 0x1F) else 2

    def on_block(uc, addr, size, ud):
        """输入：一个基本块；输出：无。预期行为：按块字节长度逐条数 Thumb 指令，累加到 steps。

        用块钩子而不是逐指令钩子：块的数量比指令少一个数量级，回调开销才扛得住。
        """
        blob = uc.mem_read(addr, size)
        i = 0
        while i < size:
            i += thumb_len(blob, i)
            steps[0] += 1

    # 钩子必须限定地址范围：不限定的话每次访存都要回 Python 一次，实测慢一个数量级。
    mu.hook_add(UC_HOOK_MEM_WRITE, on_write, begin=RCC_CR, end=RCC_CFGR + 4)
    mu.hook_add(UC_HOOK_MEM_WRITE, on_write, begin=USART1_DR, end=USART1_DR + 4)
    mu.hook_add(UC_HOOK_MEM_READ, on_read, begin=RCC_CR, end=RCC_CFGR + 4)
    mu.hook_add(UC_HOOK_MEM_READ, on_read, begin=USART1_SR, end=USART1_DR + 4)
    if dwt:
        mu.hook_add(UC_HOOK_MEM_READ, on_read, begin=DWT_CYCCNT, end=DWT_CYCCNT + 4)
    if gpio is not None:
        # OLED 的位流只走 GPIOB；CRL / CRH 也在这段地址里，一并转给钩子（它自己忽略）。
        mu.hook_add(UC_HOOK_MEM_WRITE, on_write, begin=GPIOB_BASE, end=GPIOB_END)
        mu.hook_add(UC_HOOK_MEM_READ, on_read, begin=GPIOB_IDR, end=GPIOB_IDR + 4)
    if count:
        mu.hook_add(UC_HOOK_BLOCK, on_block)

    # 复位向量：头两个字分别是栈顶与复位入口（Thumb 地址，最低位为 1）。
    mu.reg_write(UC_ARM_REG_SP, struct.unpack("<I", mu.mem_read(FLASH_BASE, 4))[0])
    pc = struct.unpack("<I", mu.mem_read(FLASH_BASE + 4, 4))[0]
    mu.reg_write(UC_ARM_REG_PC, pc)

    try:
        try:
            if not dwt:
                # PC 读回来是被掩掉 Thumb 位的偶数地址，必须补回最低位，否则 Unicorn 按 ARM 模式解码。
                mu.emu_start(pc | 1, 0, count=max_steps)
            else:
                # 分块跑：块与块之间把累计条数记进 cycles，读 CYCCNT 的钩子再把它铺进内存。
                # 为什么不用块钩子数：基本块很短，上千万次 Python 回调会把一次模拟拖到分钟级。
                done = 0
                while True:
                    if done >= max_steps:
                        raise RuntimeError("跑满 %d 条指令还没停：模型可能没在生成，或输入没被吃掉"
                                           % max_steps)
                    n = min(dwt_chunk, max_steps - done)
                    mu.emu_start(mu.reg_read(UC_ARM_REG_PC) | 1, 0, count=n)
                    done += n
                    cycles[0] = done
        except _Idle:
            pass
        except UcError as exc:
            raise RuntimeError("模拟崩了：%s（PC=0x%08x）" % (exc, mu.reg_read(UC_ARM_REG_PC)))
        else:
            raise RuntimeError("跑满 %d 条指令还没停：模型可能没在生成，或输入没被吃掉" % max_steps)
    finally:
        if was_faulthandler:
            faulthandler.enable()
    return bytes(out), (steps[0] if count else None)


def main():
    """输入：命令行；输出：固件吐出的字节。预期行为：ELF 不存在就报错提示先构建。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--elf", default=str(ROOT / "build" / "firmware" / "nanomeow.elf"))
    ap.add_argument("--text", default="你好", help="喂给 USART1 的一行内容（会自动补换行）")
    ap.add_argument("--build", action="store_true", help="先跑 build_firmware.py 再模拟")
    ap.add_argument("--count", action="store_true", help="精确统计执行的指令条数（慢）")
    args = ap.parse_args()

    if args.build:
        import subprocess
        subprocess.run([sys.executable, str(FW_DIR / "build_firmware.py")], check=True)
    elf = pathlib.Path(args.elf)
    if not elf.exists():
        raise SystemExit("找不到 %s，先跑 python infer/firmware/build_firmware.py" % elf)

    out, steps = run_firmware(elf, args.text.encode("utf-8") + b"\n", count=args.count)
    sys.stdout.buffer.write(out)
    sys.stdout.buffer.flush()
    if steps is not None:
        sys.stderr.write("\n[模拟] 执行 %d 条指令\n" % steps)


if __name__ == "__main__":
    main()
