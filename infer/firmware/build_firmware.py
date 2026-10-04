# -*- coding: utf-8 -*-
"""上板固件构建：把 infer/c 的引擎 + 8x8 字库 + OLED 显示层 + 串口入口交叉编译成
STM32F103C8T6（Cortex-M3 @72MHz，64 KiB Flash / 20 KiB RAM）可直接烧写的 ELF。

输入：infer/c 下的固件源码、infer/firmware/m3.ld、infer/firmware/arm_inc/string.h（freestanding 桩）。
输出：<build-dir>/nanomeow.elf，以及 Flash / RAM 段账本和闸门结论。
预期行为：真编译 + 真链接，不是估算。链接后按程序头统计真正要烧进 Flash 的字节，
          并在**各目标文件的未定义符号**里检查没有 __aeabi_f* / __aeabi_d*（纯整数推理）——
          注意不能在链接产物上查：lld 出来的 ELF 没有符号表，那样查永远是「无」。

用法：
    python infer/firmware/build_firmware.py
    python infer/firmware/build_firmware.py --build-dir build/firmware --clang <clang> --zig <zig>
"""

import argparse
import pathlib
import shutil
import struct
import subprocess
import sys

FW_DIR = pathlib.Path(__file__).resolve().parent
ROOT = FW_DIR.parents[1]

FLASH_BASE, FLASH_SIZE = 0x08000000, 64 * 1024
RAM_BASE, RAM_SIZE = 0x20000000, 20 * 1024

# 固件源码：引擎 + 权重表 + 生成/停止 + 字库 + OLED 协议层 + 软件 I2C + 串口入口。
# 自检程序（*_selftest.c）与主机 CLI（nm_chat.c）不上板，不在此列。
SOURCES = [
    "infer/c/nanomeow.c",
    "infer/c/nm_gen.c",
    "infer/c/nm_weights.c",
    "infer/c/nm_font.c",
    "infer/c/nm_oled.c",
    "infer/c/nm_oled_port.c",
    "infer/c/nm_fw.c",
]

INCLUDES = ["infer/c", "infer", "infer/firmware/arm_inc"]

CFLAGS = [
    "--target=armv7m-none-eabi", "-mthumb", "-mcpu=cortex-m3", "-Oz",
    "-ffreestanding", "-fno-unwind-tables", "-fno-asynchronous-unwind-tables",
    "-fno-common", "-std=c99", "-Wall", "-Wextra", "-c",
    # 每个函数 / 数据各自成段，链接期的 --gc-sections 才真的能丢没用到的代码；
    # 没有这两项时每个 .o 只有一个大 .text，--gc-sections 等于空转。
    "-ffunction-sections", "-fdata-sections",
]

# 链接期用 zig 自带的 lld（zig 自己的代码生成明显差，所以只借它的链接器）。
LINKFLAGS = ["--target=thumb-freestanding-eabi", "-mcpu=cortex_m3", "-Oz", "-Wl,--gc-sections"]

DEFAULT_CLANG = r"C:\msys64\ucrt64\bin\clang.exe"
FALLBACK_ZIG_PY = ROOT / "tmp" / "venv_zig" / "Scripts" / "python.exe"


def parse_elf(path):
    """输入：ELF 文件路径。
    输出：dict(segments=[(paddr, vaddr, filesz, memsz)], sections=[(name, size)], undef=[符号名])。
    预期行为：自己解析 ELF 头 / 程序头 / 段表 / 符号表，不依赖 objdump（x86 的 objdump 读 ARM 符号表不可靠）。"""
    data = pathlib.Path(path).read_bytes()
    if data[:4] != b"\x7fELF":
        raise ValueError("%s 不是 ELF 文件" % path)
    is64 = data[4] == 2
    endian = "<" if data[5] == 1 else ">"
    if is64:
        phoff, shoff = struct.unpack_from(endian + "QQ", data, 0x20)
        phentsize, phnum = struct.unpack_from(endian + "HH", data, 0x36)
        shentsize, shnum, shstrndx = struct.unpack_from(endian + "HHH", data, 0x3A)
    else:
        phoff, shoff = struct.unpack_from(endian + "II", data, 0x1C)
        phentsize, phnum = struct.unpack_from(endian + "HH", data, 0x2A)
        shentsize, shnum, shstrndx = struct.unpack_from(endian + "HHH", data, 0x2E)

    def shdr(i):
        off = shoff + i * shentsize
        if is64:
            return struct.unpack_from(endian + "IIQQQQIIQQ", data, off)
        return struct.unpack_from(endian + "IIIIIIIIII", data, off)

    def cstr(base, off):
        end = data.index(b"\x00", base + off)
        return data[base + off:end].decode("utf-8", "replace")

    segments = []
    for i in range(phnum):
        off = phoff + i * phentsize
        if is64:
            p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz = struct.unpack_from(
                endian + "IIQQQQQ", data, off)
        else:
            p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_flags = struct.unpack_from(
                endian + "IIIIIII", data, off)
        if p_type == 1:  # PT_LOAD
            segments.append((p_paddr, p_vaddr, p_filesz, p_memsz))

    sections, undef = [], []
    for i in range(shnum):
        name_off, s_type, _flags, _addr, _off, s_size = shdr(i)[:6]
        s_link, s_entsize = shdr(i)[6], shdr(i)[9]
        sections.append((cstr(shdr(shstrndx)[4], name_off), s_size))
        if s_type not in (2, 11):  # SHT_SYMTAB / SHT_DYNSYM
            continue
        strtab = shdr(s_link)[4]
        ent = s_entsize or (24 if is64 else 16)
        for k in range(s_size // ent):
            off = shdr(i)[4] + k * ent
            if is64:
                st_name, _info, _other, st_shndx = struct.unpack_from(endian + "IBBH", data, off)
            else:
                st_name = struct.unpack_from(endian + "I", data, off)[0]
                st_shndx = struct.unpack_from(endian + "H", data, off + 14)[0]
            if st_shndx == 0 and st_name:  # SHN_UNDEF（st_name == 0 是空名占位）
                undef.append(cstr(strtab, st_name))
    return {"segments": segments, "sections": sections, "undef": undef}


def resolve_tools(args):
    """输入：命令行参数；输出：(clang 路径, 链接器命令前缀列表)。
    预期行为：优先用 PATH 上的 clang / zig，找不到再退回本机默认位置，都没有就报错退出。"""
    clang = args.clang or shutil.which("clang") or (DEFAULT_CLANG if pathlib.Path(DEFAULT_CLANG).is_file() else None)
    if not clang:
        sys.exit("找不到 clang：用 --clang 指定交叉编译器")
    if args.zig:
        zig = [args.zig]
    elif shutil.which("zig"):
        zig = [shutil.which("zig")]
    elif FALLBACK_ZIG_PY.is_file():
        zig = [str(FALLBACK_ZIG_PY), "-m", "ziglang"]
    else:
        sys.exit("找不到 zig：用 --zig 指定，或装 pip 包 ziglang")
    return clang, zig


def main():
    """输入：无（见 --help）；输出：无。预期行为：编 + 链 + 打账本，任一闸门不过就以非 0 退出。"""
    ap = argparse.ArgumentParser(description="构建 STM32F103C8T6 上的 nanomeow 固件")
    ap.add_argument("--build-dir", type=pathlib.Path, default=ROOT / "build" / "firmware")
    ap.add_argument("--clang", default=None, help="clang 可执行文件（默认 PATH 或 msys2 ucrt64）")
    ap.add_argument("--zig", default=None, help="zig 可执行文件（默认 PATH 或 tmp/venv_zig）")
    args = ap.parse_args()

    clang, zig = resolve_tools(args)
    build_dir = args.build_dir if args.build_dir.is_absolute() else ROOT / args.build_dir
    build_dir.mkdir(parents=True, exist_ok=True)
    ld = FW_DIR / "m3.ld"
    elf = build_dir / "nanomeow.elf"

    objs = []
    for rel in SOURCES:
        src = ROOT / rel
        obj = build_dir / (src.stem + ".o")
        inc = [a for p in INCLUDES for a in ("-I", str(ROOT / p))]
        cmd = [clang] + CFLAGS + inc + [str(src), "-o", str(obj)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.exit("编译失败 %s\n%s" % (rel, r.stderr[:2000]))
        if r.stderr.strip():
            print("警告 %s：%s" % (rel, r.stderr.strip()[:400]))
        objs.append(obj)

    r = subprocess.run(zig + ["cc"] + LINKFLAGS + ["-Wl,-T," + str(ld), "-o", str(elf)] +
                       [str(o) for o in objs], capture_output=True, text=True)
    if r.returncode:
        sys.exit("链接失败\n%s" % (r.stdout + r.stderr)[:2000])

    info = parse_elf(elf)
    flash = sum(f for paddr, _v, f, _m in info["segments"] if FLASH_BASE <= paddr < FLASH_BASE + FLASH_SIZE)
    ram = sum(m for _p, vaddr, _f, m in info["segments"] if RAM_BASE <= vaddr < RAM_BASE + RAM_SIZE)

    print("--- 段账本 %s ---" % elf.name)
    for name, size in info["sections"]:
        if size:
            print("  %-16s %7d" % (name, size))
    print("Flash %d B = %.2f KiB（%d KiB 余 %d B）" % (flash, flash / 1024.0, FLASH_SIZE // 1024, FLASH_SIZE - flash))
    print("RAM   %d B = %.2f KiB（%d KiB 余 %d B）" % (ram, ram / 1024.0, RAM_SIZE // 1024, RAM_SIZE - ram))

    # 纯整数闸门：查目标文件的未定义符号，链接产物没有符号表所以查不出来。
    undef = sorted({s for o in objs for s in parse_elf(o)["undef"]})
    floats = [s for s in undef if s.startswith("__aeabi_f") or s.startswith("__aeabi_d")
              or s in ("__aeabi_lmul", "float", "double")]
    print("未定义符号：%s" % (", ".join(undef) if undef else "无"))
    print("浮点符号：%s" % ("有！%s" % floats if floats else "无（纯整数）"))

    bad = []
    if flash > FLASH_SIZE:
        bad.append("Flash 超出 %d B" % (flash - FLASH_SIZE))
    if ram > RAM_SIZE:
        bad.append("RAM 超出 %d B" % (ram - RAM_SIZE))
    if floats:
        bad.append("出现浮点符号 %s" % floats)
    if bad:
        sys.exit("闸门不过：" + "；".join(bad))
    print("闸门全过")


if __name__ == "__main__":
    main()
