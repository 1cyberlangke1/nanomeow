# -*- coding: utf-8 -*-
"""每 token 的目标 ISA「行级」动态开销估算：把「每行源码执行了几次」乘上「每行源码编出几条指令」。

和 arm_instcount.py 的分工：那个数**静态**条数（一共编出多少条指令），这个数**动态**条数
（跑一个 token 真正执行多少条）。两者共用 build_firmware.py 的 CFLAGS / SOURCES / INCLUDES，
不另抄一份口径，免得两边漂移。

输入：infer/c 下的固件源码；主机侧拿 infer/c/nm_bench.c 当驱动。
输出：stdout 的「每函数 / 每源文件」每 token 动态指令估算表，以及总量与粗估周期。
预期行为：这是**估算**，不是目标机实测：
  ① 每行执行次数来自主机 gcc -O0 --coverage 跑 nm_bench（热身 64 次 + 计时 T 次）；
     引擎全是整数运算，同一份源码在 Cortex-M3 上每行的执行次数与主机逐行相同；
  ② 每行指令数来自 clang -Oz -mcpu=cortex-m3 -g 汇编里的 .loc 归属；
  ③ 行 -> 指令的映射会被内联 / 合并 / 消除影响，所以两个口径拼出来的数字只用来比**量级与占比**。
     本机没有 ARM 模拟器（见 infer/README.md），要真数字只能上板。

用法：
    python infer/firmware/arm_linecost.py
    python infer/firmware/arm_linecost.py --tokens 4096 --top 20
"""

import argparse
import pathlib
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_firmware as bf  # noqa: E402  （同目录，借它的 CFLAGS 与源码清单）

DEFAULT_GCC = r"C:\msys64\ucrt64\bin\gcc.exe"
DEFAULT_GCOV = r"C:\msys64\ucrt64\bin\gcov.exe"

# gcov 口径只编前向路径真正会跑到的文件：OLED 显示层与上板入口（nm_fw.c）不跑，不编。
HOST_SOURCES = [
    "infer/c/nanomeow.c",
    "infer/c/nm_gen.c",
    "infer/c/nm_weights.c",
    "infer/c/nm_font.c",
    "infer/c/nm_bench.c",
]
HOST_CFLAGS = ["-std=c99", "-O0", "-g", "--coverage", "-w"]
HOST_INCLUDES = ["infer/c", "infer"]

WARMUP = 64  # nm_bench.c 计时前先跑 64 次热身（见该文件），算每 token 均值时要一起除

TYPE_RE = re.compile(r"^\s*\.type\s+([A-Za-z_][\w.$]*)\s*,\s*%function")
SIZE_RE = re.compile(r"^\s*\.size\s+([A-Za-z_][\w.$]*)\s*,")
# DWARF-5 的 `.file` 是两个字符串（目录 + 文件名），老格式只有一个；两个都要认，
# 否则文件名会取成目录名（实测踩过：全变成 "c"）。
FILE_RE = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]*)"(?:\s+"([^"]*)")?')
LOC_RE = re.compile(r"^\s*\.loc\s+(\d+)(?:\s+(\d+))?")
GCOV_SRC_RE = re.compile(r"^\s*-\s*:\s*0:Source:(.+)$")
GCOV_FUNC_RE = re.compile(r"^function\s+(\S+)\s+called\s+(\d+)")
# gcov 的次数后面可能跟一个 `*`（表示这一行里还有没跑到的基本块），必须认，
# 否则带 `*` 的行会被整行跳过（实测踩过：只剩 ##### 的行，计数看起来全是 0）。
GCOV_LINE_RE = re.compile(r"^\s*(?:(\d+)\*?|#####|=====|\$\$\$\$\$|-):\s*(\d+):")


def base(path):
    """输入：任意路径字符串；输出：文件名（统一按 / 切，Windows 反斜杠也认）。"""
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def asm_line_map(asm):
    """输入：clang -g -S 的汇编文本。
    输出：(per_line, per_func, no_loc)，键分别是 (文件名, 行号) -> 指令条数、
          {函数名: {(文件名, 行号): 条数}}、{函数名: 没有行号的指令条数}。
    预期行为：函数边界仍用 `.type X,%function` / `.size X,` 切；行号用 `.file N "路径"` 建表、
          `.loc N [行]` 推进；指令行 = 以空白开头、去空白非空、不以 `.` / `@` 开头。
          没有行号（函数序言等）的指令单独计，交给调用方按「该函数每 token 被调几次」加权。"""
    files, per_line, per_func, no_loc = {}, {}, {}, {}
    func = cur_file = cur_line = None
    for line in asm.splitlines():
        m = FILE_RE.match(line)
        if m:
            files[int(m.group(1))] = base(m.group(3) or m.group(2))
            continue
        m = TYPE_RE.match(line)
        if m:
            func = m.group(1)
            per_func.setdefault(func, {})
            continue
        if SIZE_RE.match(line):
            func = None
            continue
        m = LOC_RE.match(line)
        if m:
            n = int(m.group(1))
            cur_file = files.get(n) if n else None
            cur_line = int(m.group(2)) if (n and m.group(2)) else None
            continue
        s = line.strip()
        if not s or line[:1] not in " \t" or s.startswith(".") or s.startswith("@"):
            continue
        if func is None:
            continue
        if cur_file is None or not cur_line:
            no_loc[func] = no_loc.get(func, 0) + 1
            continue
        key = (cur_file, cur_line)
        per_line[key] = per_line.get(key, 0) + 1
        per_func[func][key] = per_func[func].get(key, 0) + 1
    return per_line, per_func, no_loc


def gcov_read(path):
    """输入：一个 .gcov 文件路径。
    输出：(per_line, calls)：键 (文件名, 行号) -> 执行次数、{函数名: 被调次数}。
    预期行为：只认 `Source:` 之后的 `次数:行号:` 记录；`#####` / `=====` / `-` 都算 0 次。
          同一个 (文件, 行号) 会在多段里重复出现（`------------------` 分段）：第一段是整份文件的
          聚合，后面每段是一个内联 / 实例拷贝，还有「本 TU 没跑到」的 `#####` 段。所以这里取**最大值**
          而不是求和 —— 求和会把同一份计数按拷贝数重复累加（实测 nm_mul 会被算成 3 倍）。"""
    per_line, calls, src = {}, {}, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = GCOV_SRC_RE.match(line)
        if m:
            src = base(m.group(1))
            continue
        m = GCOV_FUNC_RE.match(line)
        if m:
            calls[m.group(1)] = calls.get(m.group(1), 0) + int(m.group(2))
            continue
        m = GCOV_LINE_RE.match(line)
        if m and src:
            key = (src, int(m.group(2)))
            n = 0 if m.group(1) is None else int(m.group(1))
            per_line[key] = max(per_line.get(key, 0), n)
    return per_line, calls


def build_host(gcc, build_dir, tokens):
    """输入：gcc 路径、构建目录、驱动计时循环次数。
    输出：(除数, nm_bench 的输出行)。预期行为：清掉旧 .gcda（否则计数会跨次累加），
          再编译 / 链接 / 跑一次；任一失败就抛 RuntimeError。"""
    build_dir.mkdir(parents=True, exist_ok=True)
    for old in list(build_dir.glob("*.gcda")) + list(build_dir.glob("*.gcov")):
        old.unlink()
    objs = []
    for rel in HOST_SOURCES:
        src = bf.ROOT / rel
        obj = build_dir / (src.stem + ".o")
        cmd = [gcc] + HOST_CFLAGS + ["-c"]
        cmd += [a for p in HOST_INCLUDES for a in ("-I", str(bf.ROOT / p))]
        cmd += [str(src), "-o", str(obj)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("gcov 口径编译失败 %s\n%s" % (rel, r.stderr[:2000]))
        objs.append(str(obj))
    exe = build_dir / "nm_bench.exe"
    r = subprocess.run([gcc, "--coverage", "-o", str(exe)] + objs, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("gcov 口径链接失败\n%s" % r.stderr[:2000])
    r = subprocess.run([str(exe), str(tokens)], cwd=str(build_dir), capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("gcov 口径跑 nm_bench 失败\n%s" % (r.stdout + r.stderr)[:2000])
    return WARMUP + tokens, r.stdout.strip()


def run_gcov(gcov, build_dir):
    """输入：gcov 路径、构建目录；输出：生成的 .gcov 文件列表。
    预期行为：把目录里所有 .gcno 交给 gcov -f -b 跑一遍，找不到 .gcno 就报错（说明 --coverage 没生效）。"""
    notes = sorted(build_dir.glob("*.gcno"))
    if not notes:
        raise RuntimeError("没找到 .gcno：--coverage 没生效？")
    r = subprocess.run([gcov, "-f", "-b"] + [str(n) for n in notes],
                       cwd=str(build_dir), capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("gcov 失败\n%s" % (r.stdout + r.stderr)[:2000])
    return sorted(build_dir.glob("*.gcov"))


def main():
    """输入：命令行参数；输出：每函数 / 每源文件的动态指令估算表。预期行为：任何一步失败就非 0 退出。"""
    ap = argparse.ArgumentParser(description="每 token 的目标 ISA 行级动态指令估算（估算，非实测）")
    ap.add_argument("--clang", default=None, help="clang 可执行文件（默认同 build_firmware.py）")
    ap.add_argument("--gcc", default=None, help="主机 gcc（默认 PATH 或 msys2 ucrt64）")
    ap.add_argument("--gcov", default=None, help="gcov（默认 PATH 或 msys2 ucrt64）")
    ap.add_argument("--tokens", type=int, default=4096, help="驱动计时循环次数（默认 4096）")
    ap.add_argument("--build-dir", type=pathlib.Path, default=None, help="产物目录（默认 build/linecost）")
    ap.add_argument("--top", type=int, default=15, help="函数表打印前几名（默认 15）")
    args = ap.parse_args()

    build_dir = args.build_dir or (bf.ROOT / "build" / "linecost")
    if not build_dir.is_absolute():
        build_dir = bf.ROOT / build_dir

    try:
        clang, _ = bf.resolve_tools(argparse.Namespace(clang=args.clang, zig=None))
    except SystemExit:
        clang = args.clang or bf.DEFAULT_CLANG
        if not pathlib.Path(clang).is_file():
            sys.exit("找不到 clang：用 --clang 指定交叉编译器")
    gcc = args.gcc or shutil.which("gcc") or DEFAULT_GCC
    gcov = args.gcov or shutil.which("gcov") or DEFAULT_GCOV
    for name, tool in (("gcc", gcc), ("gcov", gcov)):
        if not pathlib.Path(tool).is_file():
            sys.exit("找不到 %s：用 --%s 指定" % (name, name))

    # ① 目标 ISA：每行编出几条指令
    flags = [f for f in bf.CFLAGS if f != "-c"] + ["-g", "-S", "-o", "-"]
    inc = [a for p in bf.INCLUDES for a in ("-I", str(bf.ROOT / p))]
    per_line, funcs, static_total = {}, {}, 0
    for rel in bf.SOURCES:
        r = subprocess.run([clang] + flags + inc + [str(bf.ROOT / rel)], capture_output=True, text=True)
        if r.returncode:
            sys.exit("编译失败 %s\n%s" % (rel, r.stderr[:2000]))
        pl, pf, nl = asm_line_map(r.stdout)
        for key, n in pl.items():
            per_line[key] = per_line.get(key, 0) + n
        for name, lines in pf.items():
            funcs[(rel, name)] = {"lines": lines, "no_loc": nl.get(name, 0)}
        static_total += sum(pl.values()) + sum(nl.values())

    # ② 主机执行次数
    try:
        divisor, bench = build_host(gcc, build_dir, args.tokens)
    except RuntimeError as e:
        sys.exit(str(e))
    exec_count, calls = {}, {}
    try:
        for g in run_gcov(gcov, build_dir):
            pl, cs = gcov_read(g)
            for key, n in pl.items():
                exec_count[key] = exec_count.get(key, 0) + n
            for name, n in cs.items():
                calls[name] = calls.get(name, 0) + n
    except RuntimeError as e:
        sys.exit(str(e))

    # ③ 相乘：动态条数/token = 行指令数 x 该行执行次数 / 除数，加上没行号的指令 x 函数调用次数 / 除数
    rows = []
    for (rel, name), info in funcs.items():
        dyn = sum(n * exec_count.get(key, 0) for key, n in info["lines"].items()) / float(divisor)
        dyn += info["no_loc"] * calls.get(name, 1) / float(divisor)
        if dyn > 0:
            rows.append((dyn, rel, name, sum(info["lines"].values()) + info["no_loc"]))
    rows.sort(key=lambda r: (-r[0], r[1], r[2]))
    total = sum(r[0] for r in rows)

    print("=== 每 token 目标 ISA 动态指令估算 ===")
    print("口径：执行次数 = 主机 gcc -O0 --coverage 跑 nm_bench；每行指令数 = clang -Oz -mcpu=cortex-m3 -g 的 .loc")
    print("除数 %d = 热身 %d + 计时 %d（%s）" % (divisor, WARMUP, args.tokens, bench))
    print("静态合计 %d 条指令；动态估算合计 %.0f 条指令/token（动态/静态 = %.1fx，热点在循环里反复执行）"
          % (static_total, total, total / static_total))
    print()
    print("按函数（前 %d）：" % args.top)
    for dyn, rel, name, stat in rows[:args.top]:
        print("  %-22s %-30s %9.0f 条/token（静态 %d 条）" % (rel.split("/")[-1], name, dyn, stat))
    print()
    print("按源文件：")
    per_file = {}
    for dyn, rel, _name, _stat in rows:
        per_file[rel] = per_file.get(rel, 0.0) + dyn
    for rel, dyn in sorted(per_file.items(), key=lambda kv: -kv[1]):
        print("  %-24s %9.0f 条/token（%.1f%%）" % (rel, dyn, 100.0 * dyn / total))
    print("  （只覆盖前向路径：nm_oled.c / nm_oled_port.c / nm_fw.c 不在 nm_bench 的调用链里，计 0）")
    print()
    lo, hi = total * 1.3, total * 1.8
    print("粗估周期（Cortex-M3：整数指令多数 1 周期，LDM/MLA/分支 2~3 周期，取 1.3~1.8 CPI）：")
    print("  %.0f~%.0f 周期/token -> @72 MHz 约 %.2f~%.2f ms/token（约 %.0f~%.0f tok/s）"
          % (lo, hi, lo / 72e6 * 1e3, hi / 72e6 * 1e3, 72e6 / hi, 72e6 / lo))
    print("  注意：这是**粗估**，不是板上实测 —— 真数字要上板或用周期精确模拟器。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
