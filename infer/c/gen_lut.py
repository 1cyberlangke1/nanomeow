"""生成 infer/c/nm_lut.h：把 Python 参考的 exp / log1p 两张 Q15 表原样落到 C 头文件。

输入：无（直接 import infer/ref/nonlinear.py，单一来源）。
输出：infer/c/nm_lut.h。
预期行为：表项与 Python 侧逐项相同，C 引擎与 Python 参考才不会差在查表上。
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from ref.nonlinear import EXP_LUT, EXP_LUT_Q, LOG1P_LUT  # noqa: E402


def emit(name, table, fh):
    """输入：数组名、表项列表、文件对象；输出：无。预期行为：写成 uint16_t 数组。"""
    fh.write("static const uint16_t %s[%d] = {\n" % (name, len(table)))
    for i in range(0, len(table), 12):
        fh.write("    " + ",".join(str(v) for v in table[i:i + 12]) + ",\n")
    fh.write("};\n")


def main():
    """输入：无；输出：无。预期行为：把两张表写进 infer/c/nm_lut.h。"""
    out = HERE.parents[0] / "nm_lut.h"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("/* 自动生成，请勿手改：infer/c/gen_lut.py */\n")
        fh.write("#ifndef NANOMEOW_LUT_H\n#define NANOMEOW_LUT_H\n\n")
        fh.write("#include <stdint.h>\n\n")
        fh.write("#define NM_EXP_LUT_Q %d\n\n" % EXP_LUT_Q)
        emit("nm_exp_lut", EXP_LUT, fh)
        fh.write("\n")
        emit("nm_log1p_lut", LOG1P_LUT, fh)
        fh.write("\n#endif\n")
    print("写入 %s（exp %d 项 / log1p %d 项）" % (out.name, len(EXP_LUT), len(LOG1P_LUT)))


if __name__ == "__main__":
    main()