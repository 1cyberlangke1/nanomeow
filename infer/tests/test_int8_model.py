"""整数 GEMV 与训练侧 QAT Linear 的对照测试。"""

import math
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "train"))
sys.path.insert(0, str(HERE.parents[2] / "train" / "scripts"))

from export_int8 import split_scale  # noqa: E402,F401  （口径一致性由 test_export_int8 覆盖）
from ref.fixed import normalize, value  # noqa: E402
from ref.int8_model import QMatrix, QTensor, linear  # noqa: E402
from src.qat import IntxFakeQuantizer, per_row_int8, per_tensor_int8  # noqa: E402


def to_fixed(scale):
    """输入：正浮点 scale；输出：(m, e)。预期行为：把 scale 落到定点表示上。"""
    mant, exp = math.frexp(scale)
    return normalize(int(round(mant * (1 << 31))), exp - 31)


def codes_of(tensor, scale):
    """输入：已量化的 float 张量、它的 scale；输出：整数码列表。"""
    return [int(round(float(v))) for v in (tensor / scale).reshape(-1)]


def test_linear_matches_qat_linear():
    """整数 GEMV + 动态再量化必须复现训练侧 QAT Linear 的码与 scale。"""
    torch.manual_seed(0)
    w = torch.randn(6, 4) * 0.3
    x = torch.randn(4) * 1.7

    fq_act = IntxFakeQuantizer(per_tensor_int8())
    fq_w = IntxFakeQuantizer(per_row_int8())

    xq = fq_act(x)
    s_x = float(fq_act.scale)
    wq = fq_w(w)
    s_w = [float(v) for v in fq_w.scale.reshape(-1)]
    ref = fq_act(torch.nn.functional.linear(xq, wq))
    s_ref = float(fq_act.scale)

    mat = QMatrix(
        codes_of(wq, torch.tensor(s_w).reshape(-1, 1)),
        [to_fixed(s) for s in s_w],
        w.shape[0],
        w.shape[1],
    )
    out = linear(QTensor(codes_of(xq, s_x), to_fixed(s_x)), mat)

    ref_codes = codes_of(ref, s_ref)
    assert out.codes == ref_codes, (out.codes, ref_codes)
    assert abs(value(out.scale) / s_ref - 1) < 1e-5


def test_linear_all_zero_input_is_all_zero():
    """全零输入 -> 全零输出，且 scale 不炸。"""
    mat = QMatrix([0] * 8, [normalize(1, 0)] * 2, 2, 4)
    out = linear(QTensor([0, 0, 0, 0], normalize(1, 0)), mat)
    assert out.codes == [0, 0]
