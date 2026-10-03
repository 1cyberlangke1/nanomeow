"""定点归一化（LayerNorm / GroupNorm）与训练侧 QAT 的对照测试。"""

import math
import pathlib
import sys

import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "train"))

from ref.fixed import normalize, scale_from_frac, value  # noqa: E402
from ref.int8_model import QTensor  # noqa: E402
from ref.norm import isqrt, norm_q, rsqrt_q32  # noqa: E402
from ref.nonlinear import ONE  # noqa: E402
from src.qat import IntxFakeQuantizer, per_tensor_int8  # noqa: E402


def to_fixed_scale(scale):
    """输入：正浮点 scale；输出：(m, e)。预期行为：把 scale 落到定点表示上。"""
    mant, exp = math.frexp(scale)
    return normalize(int(round(mant * (1 << 31))), exp - 31)


def codes_of(tensor, scale):
    """输入：已量化的 float 张量、它的 scale；输出：整数码列表。"""
    return [int(round(float(v))) for v in (tensor / scale).reshape(-1)]


def qt_of(tensor, scale):
    """输入：已量化的 float 张量、它的 scale；输出：对应的 QTensor。"""
    return QTensor(codes_of(tensor, scale), to_fixed_scale(scale))


def q16_of(tensor):
    """输入：实值张量；输出：直接当 Q16 码用的 QTensor（非 int8 量化的中间量）。"""
    return QTensor([int(round(float(t) * ONE)) for t in tensor.reshape(-1)],
                   scale_from_frac(16))


def assert_close_codes(got, want, min_exact=0.98):
    """输入：两串整数码；预期行为：逐位差 <= 1，且完全相同的比例不低于 min_exact。"""
    assert len(got) == len(want)
    diffs = [abs(a - b) for a, b in zip(got, want)]
    assert max(diffs) <= 1, (got, want)
    exact = sum(1 for d in diffs if d == 0) / len(diffs)
    assert exact >= min_exact, exact


def test_isqrt():
    """整数平方根必须是精确的 floor。"""
    for n in (0, 1, 2, 3, 4, 8, 15, 16, 17, 10 ** 6, 2 ** 31, 2 ** 52 - 1):
        r = isqrt(n)
        assert r * r <= n < (r + 1) * (r + 1), n


def test_rsqrt_q32_matches_math():
    """Q32 的 rsqrt vs 1/sqrt(x)：相对误差 < 1e-4。

    期望值按**量化之后**的 x 算：x 很小时 Q32 自己只有几个有效位，
    那部分误差是输入量化的，不该记在 rsqrt 的账上。
    容差再补半个 Q16 量子：输出是 Q16，1/sqrt(x) 小时它的绝对分辨率就是一个量子。
    """
    for x_real in (1e-7, 1e-5, 1e-3, 0.01, 0.5, 1.0, 4.0, 100.0, 10000.0):
        x = int(round(x_real * (1 << 32)))
        got = rsqrt_q32(x) / ONE
        want = 1.0 / math.sqrt(x / (1 << 32))
        assert abs(got - want) <= 1e-4 * want + 0.51 / ONE, (x_real, got, want)
    assert rsqrt_q32(0) == 0


def _params(seed, n=32):
    """输入：随机种子、通道数；输出：(w 的 QTensor, s_w, b 的 QTensor, s_b)。"""
    torch.manual_seed(seed)
    fq_param = IntxFakeQuantizer(per_tensor_int8())
    w = torch.randn(n) * 0.4 + 1.0
    b = torch.randn(n) * 0.2
    wq = fq_param(w)
    s_w = float(fq_param.scale)
    bq = fq_param(b)
    s_b = float(fq_param.scale)
    return qt_of(wq, s_w), qt_of(bq, s_b)


def test_layer_norm_matches_qat():
    """LayerNorm（groups=1）必须复现 fq_act(F.layer_norm(x, (C,), fq(w), fq(b), 1e-5))。"""
    w_qt, b_qt = _params(0)
    for trial in range(20):
        torch.manual_seed(100 + trial)
        fq_act = IntxFakeQuantizer(per_tensor_int8())
        x = torch.randn(32) * (0.5 + trial * 0.4)
        xq = fq_act(x)
        s_x = float(fq_act.scale)
        w = w_qt
        ref = fq_act(F.layer_norm(xq, (32,), _dequant(w_qt), _dequant(b_qt), 1e-5))
        s_ref = float(fq_act.scale)
        out = norm_q(qt_of(xq, s_x), w, b_qt, 1e-5, groups=1)
        assert_close_codes(out.codes, codes_of(ref, s_ref))
        assert abs(value(out.scale) / s_ref - 1) < 1e-3


def _dequant(qt):
    """输入：QTensor；输出：它还原出来的 float 张量。"""
    s = value(qt.scale)
    return torch.tensor([c * s for c in qt.codes])


def test_group_norm_matches_qat():
    """GroupNorm（4 组）必须复现 fq_act(F.group_norm(x, 4, fq(w), fq(b), 64e-5))。"""
    w_qt, b_qt = _params(7)
    for trial in range(20):
        torch.manual_seed(200 + trial)
        fq_act = IntxFakeQuantizer(per_tensor_int8())
        x = torch.randn(32) * (0.5 + trial * 0.4)
        xq = fq_act(x)
        s_x = float(fq_act.scale)
        ref = fq_act(F.group_norm(xq.view(1, 32), 4, _dequant(w_qt), _dequant(b_qt), 64e-5))
        s_ref = float(fq_act.scale)
        out = norm_q(qt_of(xq, s_x), w_qt, b_qt, 64e-5, groups=4)
        assert_close_codes(out.codes, codes_of(ref, s_ref))
        assert abs(value(out.scale) / s_ref - 1) < 1e-3


def test_norm_q_accepts_non_int8_scale():
    """wkv 的输出不是 int8 量化的：用 Q16 的码直接喂进去也必须和 float 版一致。"""
    w_qt, b_qt = _params(3)
    torch.manual_seed(9)
    fq_act = IntxFakeQuantizer(per_tensor_int8())
    x = torch.randn(32) * 4.0
    ref = fq_act(F.layer_norm(x, (32,), _dequant(w_qt), _dequant(b_qt), 1e-5))
    s_ref = float(fq_act.scale)
    out = norm_q(q16_of(x), w_qt, b_qt, 1e-5)
    assert_close_codes(out.codes, codes_of(ref, s_ref))
    assert abs(value(out.scale) / s_ref - 1) < 1e-3


def test_norm_q_small_variance_eps_matters():
    """小方差（eps 量级）：eps 参与与否会改变结果，且必须和 float 版一致。

    偏差取 1e-3 量级（约 65 个 Q16 量子）。再小就撞上 Q16 的分辨率极限了：
    偏差只剩几个量子时，均值的半个量子舍入误差会被 rsqrt(eps) 放大到能翻码，
    那是定点精度的物理上限，不是实现 bug。
    """
    w_qt, b_qt = _params(11)
    torch.manual_seed(12)
    fq_act = IntxFakeQuantizer(per_tensor_int8())
    x = torch.tensor([0.5 + 1e-3 * i for i in range(32)])
    ref = fq_act(F.layer_norm(x, (32,), _dequant(w_qt), _dequant(b_qt), 1e-5))
    s_ref = float(fq_act.scale)
    out = norm_q(q16_of(x), w_qt, b_qt, 1e-5)
    assert_close_codes(out.codes, codes_of(ref, s_ref))
    assert abs(value(out.scale) / s_ref - 1) < 1e-3


def test_norm_q_constant_input_is_zero():
    """全等输入：分子恒为 0，输出必须全 0，不能因为 rsqrt(eps) 变大而爆掉。"""
    w_qt, b_qt = _params(5)
    zero_b = QTensor([0] * 32, b_qt.scale)
    x = q16_of(torch.full((32,), 0.7))
    out = norm_q(x, w_qt, zero_b, 1e-5)
    assert out.codes == [0] * 32
