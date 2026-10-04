"""生成 infer/c/nm_lut.h：把 Python 参考的 exp / log1p 两张 Q15 表编码成 C 头文件。

输入：无（直接 import infer/ref/nonlinear.py，单一来源）。
输出：infer/c/nm_lut.h。
预期行为：两张表不原样存，改存「初值 + 首差 + 每步 2 bit 二阶差」位流，C 侧 nm_lut_init()
          纯整数展开回原表。本脚本自带反解自检：解出来的表必须与 Python 侧逐项相同，
          二阶差跨度装不进 2 bit 时直接报错退出，绝不静默截断。
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from ref.nonlinear import EXP_LUT, EXP_LUT_Q, LOG1P_LUT  # noqa: E402

CODE_BITS = 2       # 每个二阶差占几个 bit
WORDS_PER_ROW = 12  # 位流每行打几个 uint32


def encode(table):
    """输入：Q15 表（长度 NM_LUT_N）；输出：(v0, d0, lo, codes)。
    预期行为：一阶差 d[i] = t[i+1] - t[i]、二阶差 dd[i] = d[i+1] - d[i] 必须全部落在
              [lo, lo + 2**CODE_BITS - 1] 里，否则抛异常（编码装不下）。"""
    d0 = table[1] - table[0]
    d = d0
    dd = []
    for i in range(1, len(table) - 1):
        nxt = table[i + 1] - table[i]
        dd.append(nxt - d)
        d = nxt
    lo, hi = min(dd), max(dd)
    if hi - lo >= (1 << CODE_BITS):
        raise ValueError("二阶差跨度 %d..%d 超过 %d 个码位" % (lo, hi, 1 << CODE_BITS))
    return table[0], d0, lo, [x - lo for x in dd]


def pack(codes):
    """输入：码位列表（每个 < 2**CODE_BITS）；输出：uint32 列表。
    预期行为：第 k 个码位放在第 k>>4 个字的 (k&15)*CODE_BITS 位上，与 C 侧解码同序。"""
    words = [0] * ((len(codes) + 15) // 16)
    for k, c in enumerate(codes):
        words[k >> 4] |= c << ((k & 15) * CODE_BITS)
    return words


def decode(words, n, v0, d0, lo):
    """输入：位流、表长、初值 / 首差 / 二阶差基值；输出：解出来的表。
    预期行为：与 C 侧 nm_lut_expand 同一套递推，供本脚本自检。"""
    out, v, d = [v0], v0, d0
    for i in range(1, n):
        if i > 1:  # 第 1 步用的还是首差 d0，从第 2 步起才吃第一个二阶差字段
            k = i - 2
            d += lo + ((words[k >> 4] >> ((k & 15) * CODE_BITS)) & ((1 << CODE_BITS) - 1))
        v += d
        out.append(v)
    return out


def emit(name, table, fh):
    """输入：表名前缀、Q15 表、文件对象；输出：位流字节数。
    预期行为：写出 v0 / d0 / lo / 位流四样，并断言反解结果与输入逐项相同。"""
    v0, d0, lo, codes = encode(table)
    words = pack(codes)
    assert decode(words, len(table), v0, d0, lo) == list(table), "%s 反解自检失败" % name
    fh.write("static const uint16_t %s_v0 = %d;\n" % (name, v0))
    fh.write("static const int16_t %s_d0 = %d;\n" % (name, d0))
    fh.write("static const int8_t %s_lo = %d;\n" % (name, lo))
    fh.write("static const uint32_t %s_dd[%d] = {\n" % (name, len(words)))
    for i in range(0, len(words), WORDS_PER_ROW):
        fh.write("    " + ",".join("0x%08xu" % w for w in words[i:i + WORDS_PER_ROW]) + ",\n")
    fh.write("};\n")
    return len(words) * 4 + 5  # 位流 + v0/d0/lo 各一份


def main():
    """输入：无；输出：无。预期行为：把两张表编码进 infer/c/nm_lut.h，并打印省下的字节数。"""
    n = len(EXP_LUT)
    assert len(LOG1P_LUT) == n, "两张表长度必须一致"
    nanomeow_h = (HERE.parents[0] / "nanomeow.h").read_text(encoding="utf-8")
    want = "#define NM_LUT_N %d" % n
    if want not in nanomeow_h:
        raise SystemExit("infer/c/nanomeow.h 里缺 `%s`，两边表长会不一致" % want)
    out = HERE.parents[0] / "nm_lut.h"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("/* 自动生成，请勿手改：infer/c/gen_lut.py\n")
        fh.write(" *\n")
        fh.write(" * 两张 Q15 表不原样存，存的是「初值 v0 + 首差 d0 + 每步 2 bit 二阶差」位流：\n")
        fh.write(" *   v[0] = v0;  d[0] = d0;\n")
        fh.write(" *   v[i+1] = v[i] + d[i];  d[i+1] = d[i] + (lo + 位流里第 i 个 2 bit 字段)\n")
        fh.write(" * 表是平滑的（exp 的二阶差只落在 -1..2、log1p 只落在 -2..1），所以 2 bit 够用。\n")
        fh.write(" * 展开是纯整数、逐项与 infer/ref/nonlinear.py 相同（gen_lut.py 里有反解自检）。\n")
        fh.write(" */\n")
        fh.write("#ifndef NANOMEOW_LUT_H\n#define NANOMEOW_LUT_H\n\n")
        fh.write("#include <stdint.h>\n\n")
        fh.write("#ifndef NM_LUT_N\n#define NM_LUT_N %d\n#endif\n" % n)
        fh.write("#define NM_EXP_LUT_Q %d\n\n" % EXP_LUT_Q)
        exp_bytes = emit("nm_exp_lut", EXP_LUT, fh)
        fh.write("\n")
        log_bytes = emit("nm_log1p_lut", LOG1P_LUT, fh)
        fh.write("\n#endif\n")
    raw = 2 * n * 2
    print("写入 %s：%d 项 x2 原样 %d B -> 位流 %d B（省 %d B）"
          % (out.name, n, raw, exp_bytes + log_bytes, raw - exp_bytes - log_bytes))


if __name__ == "__main__":
    main()
