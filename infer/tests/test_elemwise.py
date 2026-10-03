"""逐元素算子的对照测试：定点结果必须贴着 float32 参考，误差不超过输出量化的一格。

参照物是「输入经过同一套 per-tensor 假量化后，用 float32 算同一条式子」——输入侧的
量化误差两边共有，所以剩下的差只可能来自定点中间量，容差按输出的一格量化步长算。
"""

import math
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "train"))

from ref.elemwise import (  # noqa: E402
    qt_add,
    qt_add_real,
    qt_addmul,
    qt_mul,
    qt_mul_real,
    qt_neg,
    qt_relu_sq,
    qt_sub,
    qt_sum_head,
)
from ref.fixed import normalize, value  # noqa: E402
from ref.int8_model import QTensor  # noqa: E402
from src.qat import IntxFakeQuantizer, per_tensor_int8  # noqa: E402

FRAC = 1.0 / 65536


def to_fixed_scale(scale):
    """输入：正浮点 scale；输出：(m, e)。预期行为：把 scale 落到定点表示上。"""
    mant, exp = math.frexp(scale)
    return normalize(int(round(mant * (1 << 31))), exp - 31)


def qt(tensor):
    """输入：float 张量；输出：(QTensor, 反量化后的 float 张量, 量化步长)。"""
    fq = IntxFakeQuantizer(per_tensor_int8())
    out = fq(tensor)
    s = float(fq.scale)
    codes = [int(round(float(v) / s)) for v in out.reshape(-1)]
    return QTensor(codes, to_fixed_scale(s)), out, s


def deq(qtensor):
    """输入：QTensor；输出：反量化后的 float 列表。"""
    v = value(qtensor.scale)
    return [c * v for c in qtensor.codes]


def check(got, ref, scale=None, steps=1.5):
    """输入：定点结果、float32 参考、比较用的 scale（None = 参考自己量化后的步长）。

    预期行为：逐元素差不超过 steps 个 scale；steps 默认 1.5 覆盖「输出被量化过一次」
              （半格）加中间量的定点舍入。
    """
    if scale is None:
        _, _, scale = qt(ref)
    err = max(abs(a - b) for a, b in zip(deq(got), ref.reshape(-1).tolist()))
    assert err <= steps * scale, (err, scale)


def pair(seed, n=64, a_scale=1.5, b_scale=0.7):
    """输入：种子与两路幅度；输出：(qa, qb, 反量化 a, 反量化 b)。"""
    torch.manual_seed(seed)
    qa, da, _ = qt(torch.randn(n) * a_scale)
    qb, db, _ = qt(torch.randn(n) * b_scale)
    return qa, qb, da, db


def test_qt_add_and_sub():
    """加减法：允许两边 scale 不同，结果要贴住 float32 的和 / 差。"""
    qa, qb, da, db = pair(1)
    check(qt_add(qa, qb), da + db)
    check(qt_sub(qa, qb), da - db)


def test_qt_mul_quantized():
    """乘法（量化版）：对应训练侧 `fq_act(k * k_k)`。"""
    qa, qb, da, db = pair(2)
    check(qt_mul(qa, qb), da * db)


def test_qt_mul_unquantized_keeps_product_scale():
    """乘法（不量化）：scale 必须是两边之积，误差只有一个定点量子。"""
    qa, qb, da, db = pair(3)
    got = qt_mul(qa, qb, quantize=False)
    assert abs(value(got.scale) - value(qa.scale) * value(qb.scale)) < 1e-12
    check(got, da * db, scale=value(got.scale), steps=2)


def test_qt_add_real_and_mul_real():
    """张量加常数 / 乘有理数：对应 `- softplus(...) - 0.5` 与 `a - 1`。"""
    qa, _, da, _ = pair(4)
    check(qt_add_real(qa, -1, 2), da - 0.5)
    check(qt_mul_real(qa, 3, 2), da * 1.5)


def test_qt_addmul():
    """a + b * c：对应 time-shift 混合与 value residual 的 lerp。"""
    qa, qb, da, db = pair(5)
    qc, dc, _ = qt(torch.randn(64) * 0.3)
    check(qt_addmul(qa, qb, qc), da + db * dc)


def test_qt_relu_sq():
    """relu 后平方：对应 CMix 的 `fq_act(relu(key(k)) ** 2)`。"""
    torch.manual_seed(6)
    qa, da, _ = qt(torch.randn(64) * 1.2)
    check(qt_relu_sq(qa), torch.relu(da) ** 2)


def test_qt_neg():
    """取负：码逐元素取负、scale 不变（对应训练侧的 `-kk`）。"""
    qa, _, _, _ = pair(7)
    got = qt_neg(qa)
    assert got.codes == [-c for c in qa.codes]
    assert got.scale == qa.scale


def test_qt_sum_head():
    """head 内求和并广播：对应 `(r * k * r_k).sum(dim=-1, keepdim=True)`。"""
    torch.manual_seed(8)
    qa, da, _ = qt(torch.randn(32) * 0.8)
    ref = da.view(4, 8).sum(dim=-1, keepdim=True).expand(4, 8).reshape(-1)
    got = qt_sum_head(qa, 8)
    assert len(got.codes) == 32
    check(got, ref, scale=value(got.scale), steps=2)
