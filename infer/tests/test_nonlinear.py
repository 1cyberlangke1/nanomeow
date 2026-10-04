"""定点非线性（exp / log1p 双 LUT + sigmoid / tanh / softplus）与训练侧 QAT 的对照测试。"""

import math
import pathlib
import sys

import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "train"))

from ref.fixed import normalize, rescale, scale_from_frac, value  # noqa: E402
from ref.int8_model import QTensor  # noqa: E402
from ref.nonlinear import (  # noqa: E402
    EXP_LUT,
    LOG1P_LUT,
    ONE,
    exp_q,
    log1p_q,
    qt_exp,
    qt_sigmoid,
    qt_softplus,
    qt_tanh,
    sigmoid_q,
    softplus_q,
    tanh_q,
)
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


def fx(v):
    """输入：实值；输出：Q(FRAC_BITS) 整数。"""
    return int(round(v * ONE))


def assert_close_codes(got, want, min_exact=0.98):
    """输入：两串整数码；预期行为：逐位差 <= 1，且完全相同的比例不低于 min_exact。"""
    assert len(got) == len(want)
    diffs = [abs(a - b) for a, b in zip(got, want)]
    assert max(diffs) <= 1, (got, want)
    exact = sum(1 for d in diffs if d == 0) / len(diffs)
    assert exact >= min_exact, exact


def assert_close_codes_many(pairs, min_exact=0.98):
    """输入：(got, want) 码对列表；预期行为：整体逐位差 <= 1，总命中率不低于 min_exact。

    单组 128 个样本的命中率自身就有约 0.6% 的抽样抖动（真实命中率 ~99.5%），
    逐组卡 0.98 等于掷骰子；把 20 组（2560 个样本）合起来统计，阈值才有统计意义。
    """
    diffs = [abs(a - b) for got, want in pairs for a, b in zip(got, want)]
    assert max(diffs) <= 1
    exact = sum(1 for d in diffs if d == 0) / len(diffs)
    assert exact >= min_exact, exact


def test_rescale_round_trip():
    """rescale 落到 Q(FRAC_BITS) 再乘回来，误差必须在一个量子以内。"""
    for scale in (0.5, 1.0 / 127, 3.7e-3, 2.0 ** -20, 1.0):
        s = to_fixed_scale(scale)
        for code in (-128, -77, -1, 0, 1, 63, 127):
            back = rescale(code, s, 16) * value(scale_from_frac(16))
            assert abs(back - code * scale) <= 0.5 / ONE, (scale, code, back)


def test_lut_endpoints_and_monotonicity():
    """两张表的端点必须是精确值，且各自单调不减。"""
    assert EXP_LUT[0] == 0
    assert EXP_LUT[256] == 1 << 15
    assert LOG1P_LUT[0] == 0
    assert LOG1P_LUT[256] == int(round(math.log(2) * (1 << 15)))
    assert all(b >= a for a, b in zip(EXP_LUT, EXP_LUT[1:]))
    assert all(b >= a for a, b in zip(LOG1P_LUT, LOG1P_LUT[1:]))


def test_exp_q_matches_math():
    """exp 定点 vs math.exp：相对误差 < 1e-3，另加一个 Q16 量子的绝对下限。

    下限是必要的：x 很负时 exp(x) 只有几个量子，Q16 的输出分辨率本身就是瓶颈，
    不是表或插值的误差。
    """
    for i in range(-1100, 1, 7):
        x = i / 100.0
        got = exp_q(fx(x)) / ONE
        want = math.exp(x)
        assert abs(got - want) <= 1e-3 * want + 1.5e-5, (x, got, want)


def test_exp_q_saturates():
    """exp 的边界行为：>= 0 归 1，< -17 * ln2 向零饱和。"""
    assert exp_q(0) == ONE
    assert exp_q(5 * ONE) == ONE
    assert exp_q(-12 * ONE) == 0
    assert exp_q(-100 * ONE) == 0
    assert exp_q(-11 * ONE) > 0


def test_sigmoid_q_matches_math():
    """sigmoid 定点 vs 1/(1+exp(-x))：绝对误差 < 1e-4，且两端精确饱和。"""
    for i in range(-1200, 1201, 13):
        x = i / 100.0
        got = sigmoid_q(fx(x)) / ONE
        want = 1.0 / (1.0 + math.exp(-x))
        assert abs(got - want) < 1e-4, (x, got, want)
    assert sigmoid_q(0) == ONE // 2
    assert sigmoid_q(40 * ONE) == ONE
    assert sigmoid_q(-40 * ONE) == 0


def test_tanh_q_matches_math():
    """tanh 定点 vs math.tanh：绝对误差 < 1e-4。"""
    for i in range(-800, 801, 11):
        x = i / 100.0
        got = tanh_q(fx(x)) / ONE
        want = math.tanh(x)
        assert abs(got - want) < 1e-4, (x, got, want)
    assert tanh_q(0) == 0
    assert tanh_q(40 * ONE) == ONE
    assert tanh_q(-40 * ONE) == -ONE


def test_softplus_q_matches_math():
    """softplus 定点 vs log(1+exp(x))：绝对误差 < 1e-4。"""
    for i in range(-1200, 1201, 17):
        x = i / 100.0
        got = softplus_q(fx(x)) / ONE
        want = math.log1p(math.exp(x))
        assert abs(got - want) < 1e-4, (x, got, want)
    assert softplus_q(0) == log1p_q(ONE)
    assert softplus_q(-100 * ONE) == 0
    assert softplus_q(100 * ONE) == 100 * ONE


def test_log1p_q_matches_math():
    """log1p 定点 vs math.log1p：绝对误差 < 1e-4。"""
    for i in range(0, 1001, 7):
        t = i / 1000.0
        got = log1p_q(fx(t)) / ONE
        want = math.log1p(t)
        assert abs(got - want) < 1e-4, (t, got, want)


def _qat_pairs(kind, ref_fn, count=128, spread=3.0, trials=20):
    """输入：非线性名、torch 参考函数；输出：逐组 (QTensor 输入, QAT 输出码, QAT scale)。

    预期行为：抽样用 kind 派生的固定种子，同一测试每次跑到的样本完全一样。
              不设种子时 20 组里偶尔会抽到一组让命中率掉到阈值以下，测试就变成掷骰子。
    """
    torch.manual_seed(20261004 + sum(kind.encode("utf-8")))
    for _ in range(trials):
        fq = IntxFakeQuantizer(per_tensor_int8())
        z = torch.randn(count) * spread
        zq = fq(z)
        s_in = float(fq.scale)
        ref = fq(ref_fn(zq))
        yield qt_of(zq, s_in), codes_of(ref, float(fq.scale)), float(fq.scale)


def test_qt_sigmoid_matches_qat():
    """qt_sigmoid 必须复现训练侧 fq_act(sigmoid(x)) 的 int8 码与 scale。"""
    pairs = []
    for qt_in, want, s_ref in _qat_pairs("sigmoid", torch.sigmoid):
        out = qt_sigmoid(qt_in)
        assert abs(value(out.scale) / s_ref - 1) < 1e-3
        pairs.append((out.codes, want))
    assert_close_codes_many(pairs)


def test_qt_tanh_matches_qat():
    """qt_tanh 必须复现训练侧 fq_act(tanh(x)) 的 int8 码与 scale。"""
    pairs = []
    for qt_in, want, s_ref in _qat_pairs("tanh", torch.tanh):
        out = qt_tanh(qt_in)
        assert abs(value(out.scale) / s_ref - 1) < 1e-3
        pairs.append((out.codes, want))
    assert_close_codes_many(pairs)


def test_qt_softplus_matches_qat():
    """qt_softplus 必须复现训练侧 fq_act(softplus(x)) 的 int8 码与 scale。"""
    pairs = []
    for qt_in, want, s_ref in _qat_pairs("softplus", F.softplus, spread=2.0):
        out = qt_softplus(qt_in)
        assert abs(value(out.scale) / s_ref - 1) < 1e-3
        pairs.append((out.codes, want))
    assert_close_codes_many(pairs)


def test_qt_exp_matches_qat():
    """qt_exp 必须复现训练侧 fq_act(exp(x)) 的 int8 码（输入限定非正）。"""
    torch.manual_seed(1)
    pairs = []
    for _ in range(20):
        fq = IntxFakeQuantizer(per_tensor_int8())
        z = -torch.rand(128) * 3.0
        zq = fq(z)
        s_in = float(fq.scale)
        ref = fq(torch.exp(zq))
        out = qt_exp(qt_of(zq, s_in))
        assert abs(value(out.scale) / float(fq.scale) - 1) < 1e-3
        pairs.append((out.codes, codes_of(ref, float(fq.scale))))
    assert_close_codes_many(pairs)


def test_qt_ops_all_zero_input():
    """全零输入：exp 给 1、sigmoid 给 0.5、tanh 给 0、softplus 给 ln2。"""
    zero = QTensor([0] * 8, to_fixed_scale(1.0))
    assert qt_exp(zero).codes == [127] * 8
    assert qt_sigmoid(zero).codes == [127] * 8 or qt_sigmoid(zero).codes == [0] * 8
    assert qt_tanh(zero).codes == [0] * 8
    assert qt_softplus(zero).codes == [127] * 8
