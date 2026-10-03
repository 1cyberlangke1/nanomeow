"""定点工具测试：归一化、舍入、scale 算术与动态量化。"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ref.fixed import (  # noqa: E402
    MANT_MIN,
    QMAX,
    apply_scale,
    div,
    div_int,
    mul,
    mul_int,
    normalize,
    quantize_dynamic,
    ratio,
    round_div,
    value,
)


def test_round_div_is_half_to_even():
    """中点位取偶，与 torch.round / Python round 一致。"""
    assert round_div(5, 2) == 2
    assert round_div(7, 2) == 4
    assert round_div(-5, 2) == -2
    assert round_div(-7, 2) == -4
    assert round_div(10, 5) == 2


def test_normalize_keeps_value_in_range():
    """归一化只挪指数：乘子必须落在 [2^30, 2^31)，值不变。"""
    for m, e in ((1, 0), (3, -7), (2 ** 40, 5), (MANT_MIN, 0), (MANT_MIN - 1, 11)):
        nm, ne = normalize(m, e)
        assert MANT_MIN <= nm <= (1 << 31) - 1
        assert nm * (2.0 ** ne) == m * (2.0 ** e)


def test_mul_div_roundtrip():
    """乘完再除回来，相对误差要在定点精度内。"""
    a = normalize(12345, -3)
    b = normalize(7, 4)
    assert abs(value(div(mul(a, b), b)) / value(a) - 1) < 1e-9
    assert abs(value(div_int(mul_int(a, 9), 9)) / value(a) - 1) < 1e-9


def test_ratio_and_apply():
    """a / b 的比值用 scale 表示后，缩放一个 int32 要准。"""
    assert apply_scale(900, ratio(normalize(1, 0), normalize(3, 0))) == 300


def test_mul_div_are_not_off_by_2_pow_30():
    """直接对值：乘/除的指数不能漏掉 2^30（往返测试抓不到这个错）。"""
    a = normalize(1, 0)
    b = normalize(1, 0)
    assert abs(value(mul(a, b)) - 1.0) < 1e-9
    assert abs(value(div(a, b)) - 1.0) < 1e-9
    assert abs(value(mul(normalize(1, 0), normalize(1, -20))) - 2.0 ** -20) < 1e-20


def test_quantize_dynamic_matches_no_clipping_err():
    """正负端各除自己的边界取大者：这里负端更大，步长按 |min|/128 走。"""
    codes = [3, -100, 55, 0]
    q, new_scale = quantize_dynamic(codes, normalize(1, 0))
    assert q == [4, -128, 70, 0]
    assert abs(value(new_scale) / (100.0 / 128.0) - 1) < 1e-9


def test_quantize_dynamic_positive_only_uses_127():
    """全是非负数时负端为 0，步长按 max/127 走。"""
    codes = [0, 10, 50]
    q, new_scale = quantize_dynamic(codes, normalize(1, 0))
    assert q == [0, 25, 127]
    assert abs(value(new_scale) / (50.0 / 127.0) - 1) < 1e-9


def test_quantize_dynamic_is_scale_invariant_and_bounded():
    """码只取决于张量内部的相对形状：换 scale 结果一样，且落在 int8 范围内。"""
    codes = [3, -100, 55, 0]
    ref = None
    for scale in (normalize(1, 0), normalize(1, -20), normalize(5, 7)):
        q, _ = quantize_dynamic(codes, scale)
        assert all(-128 <= v <= 127 for v in q)
        assert max(abs(v) for v in q) in (QMAX, QMAX + 1)
        if ref is None:
            ref = q
        assert q == ref


def test_quantize_dynamic_all_zero_and_idempotent():
    """全零张量原样返回；已经量化过的码再量化必须逐位不动。"""
    zero_scale = normalize(1, 0)
    assert quantize_dynamic([0, 0, 0], zero_scale) == ([0, 0, 0], zero_scale)
    codes = [127, -128, 64, -64]
    assert quantize_dynamic(codes, zero_scale)[0] == codes
