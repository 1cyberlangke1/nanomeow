# -*- coding: utf-8 -*-
"""Cortex-M3 侧的静态指令条数账本：把引擎各源文件用**与上板完全相同**的 CFLAGS 编成汇编，
再数每个函数里真正的指令行。用来判断一次优化有没有让目标 ISA 的代码变胖 ——
主机基准量的是执行时间，这个量的是代码体积，两个一起看才完整。

输入：infer/c 下的固件源码；编译参数直接 import infer/firmware/build_firmware.py 的
      CFLAGS / SOURCES / INCLUDES（唯一来源，不另抄一份，免得两边漂移）。
输出：stdout 的每函数指令条数表（按条数降序）与各目标文件合计。
预期行为：这是**静态**账本 —— 数的是「有多少条指令」，不是「每条跑几次」。
          clang -S 的输出里函数由 `.type <名>,%function` 起、`.size <名>,` 止，
          中间的指令行以空白开头且不以 `.` / `@` 开头；用这两个标记切函数，不靠名字猜。
          注意 -ffunction-sections 下还会出现 `OUTLINED_FUNCTION_n` 这种编译器拆出来的冷块，
          它们也是真代码，一并计入。

用法：
    python infer/firmware/arm_instcount.py
    python infer/firmware/arm_instcount.py --clang <clang>
"""

import argparse
import re
import subprocess
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_firmware as bf  # noqa: E402  （同目录，借它的 CFLAGS 与源码清单）

TYPE_RE = re.compile(r"^\s*\.type\s+([A-Za-z_][\w.$]*)\s*,\s*%function")
SIZE_RE = re.compile(r"^\s*\.size\s+([A-Za-z_][\w.$]*)\s*,")


def count_functions(asm):
    """输入：clang -S 的汇编文本；输出：{函数名: 指令条数}。

    预期行为：只在 `.type X,%function` 与 `.size X,` 之间计数；指令行 = 以空白开头、
    去掉首尾空白后非空、且不以 `.`（伪指令）/ `@`（注释）开头。找不到边界就跳过，
    绝不靠「名字像函数」来猜。
    """
    counts, cur, n = {}, None, 0
    for line in asm.splitlines():
        m = TYPE_RE.match(line)
        if m:
            cur, n = m.group(1), 0
            continue
        if SIZE_RE.match(line):
            if cur is not None:
                counts[cur] = counts.get(cur, 0) + n
            cur = None
            continue
        if cur is None:
            continue
        s = line.strip()
        if s and line[:1] in " \t" and not s.startswith(".") and not s.startswith("@"):
            n += 1
    return counts


def main():
    """输入：命令行参数；输出：每函数指令条数表。预期行为：任一源文件编不过就非 0 退出。"""
    ap = argparse.ArgumentParser(description="Cortex-M3 静态指令条数账本")
    ap.add_argument("--clang", default=None, help="clang 可执行文件（默认同 build_firmware.py）")
    ap.add_argument("--top", type=int, default=12, help="每个目标文件打印前几名（默认 12）")
    args = ap.parse_args()

    try:
        clang, _ = bf.resolve_tools(argparse.Namespace(clang=args.clang, zig=None))
    except SystemExit:
        # 本工具只编不链，缺 zig 不该拦住它：退回 clang 的默认位置（口径与 build_firmware 一致）
        clang = args.clang or bf.DEFAULT_CLANG
        if not pathlib.Path(clang).is_file():
            sys.exit("找不到 clang：用 --clang 指定交叉编译器")
    flags = [f for f in bf.CFLAGS if f != "-c"] + ["-S", "-o", "-"]
    inc = [a for p in bf.INCLUDES for a in ("-I", str(bf.ROOT / p))]

    grand = 0
    for rel in bf.SOURCES:
        r = subprocess.run([clang] + flags + inc + [str(bf.ROOT / rel)],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit("编译失败 %s\n%s" % (rel, r.stderr[:2000]))
        counts = count_functions(r.stdout)
        total = sum(counts.values())
        grand += total
        print("=== %s  合计 %d 条 ===" % (rel, total))
        for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:args.top]:
            print("  %-34s %5d" % (name, n))
    print("=== 全部 %d 个目标文件合计 %d 条指令 ===" % (len(bf.SOURCES), grand))
    return 0


if __name__ == "__main__":
    sys.exit(main())