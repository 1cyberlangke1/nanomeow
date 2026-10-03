"""逐元素算子：训练侧被 `fq_act(...)` 包住的加法 / 乘法 / 混合，在推理侧落成整数运算。

输入：QTensor（整数码 + 一个 scale）。
输出：QTensor。
预期行为：中间量统一走 Q(FRAC_BITS) 定点（`to_fixed` / `from_fixed`），末尾的
          `from_fixed` 就是训练侧的 `fq_act`（per-tensor 动态量化）；`quantize=False`
          的版本不量化、保留中间量的 scale，对应训练侧「乘积后面还要过别的算子、
          还没到 fq_act 点」的位置。所有乘法的中间量在 C 侧要 int64。
"""

from .fixed import mul, quantize_dynamic, round_div, scale_from_frac
from .int8_model import QTensor
from .nonlinear import FRAC_BITS, ONE, from_fixed, to_fixed


def _finish(vals, quantize):
    """输入：Q(FRAC_BITS) 整数列表、是否量化；输出：QTensor。

    预期行为：quantize=True 走 per-tensor 动态量化（= 训练侧 fq_act）；
              False 时保留 scale = 2^-FRAC_BITS。
    """
    if quantize:
        return from_fixed(vals)
    return QTensor(vals, scale_from_frac(FRAC_BITS))


def qt_mul(x, y, quantize=True):
    """输入：两个同长 QTensor（逐元素相乘）；输出：QTensor。

    预期行为：码逐元素相乘、scale 相乘；quantize=True 时再走一次 per-tensor 动态量化
              （对应训练侧 `fq_act(k * k_k)`），False 时保留乘积 scale
              （对应 `kk * a` 这种后面还要进 wkv7 再量化的中间量）。
    """
    assert len(x.codes) == len(y.codes), "两个 QTensor 长度必须相同"
    codes = [a * b for a, b in zip(x.codes, y.codes)]
    s = mul(x.scale, y.scale)
    if quantize:
        q, s = quantize_dynamic(codes, s)
        return QTensor(q, s)
    return QTensor(codes, s)


def qt_neg(x):
    """输入：QTensor；输出：逐元素取负的 QTensor（scale 不变）。

    预期行为：对应训练侧 `-kk`；码取负后仍在 [-128, 127] 内（参考实现里不会出现
              -(-128)，因为 kk 是 normalize 的输出）。
    """
    return QTensor([-c for c in x.codes], x.scale)


def qt_add(a, b, quantize=True):
    """输入：两个同长 QTensor；输出：逐元素相加的 QTensor。

    预期行为：两边各自落到 Q(FRAC_BITS) 再相加，所以允许 scale 不同；
              对应训练侧的残差相加 `x + attn_out`。
    """
    va, vb = to_fixed(a), to_fixed(b)
    return _finish([p + q for p, q in zip(va, vb)], quantize)


def qt_sub(a, b, quantize=True):
    """输入：两个同长 QTensor；输出：a - b。预期行为：对应 time-shift 的 `prev - x`。"""
    va, vb = to_fixed(a), to_fixed(b)
    return _finish([p - q for p, q in zip(va, vb)], quantize)


def qt_add_real(a, num, den=1, quantize=True):
    """输入：QTensor、有理数 num/den；输出：a + num/den。

    预期行为：对应训练侧的 `-softplus(...) - 0.5`、`a - 1` 这类「张量加常数」。
    """
    c = round_div(num * ONE, den)
    return _finish([v + c for v in to_fixed(a)], quantize)


def qt_mul_real(a, num, den=1, quantize=True):
    """输入：QTensor、有理数 num/den；输出：a * num/den。"""
    return _finish([round_div(v * num, den) for v in to_fixed(a)], quantize)


def qt_addmul(a, b, c, quantize=True):
    """输入：三个同长 QTensor；输出：a + b * c。

    预期行为：对应训练侧的 `x + xx * p`（time-shift 混合）与
              `v + (v_first - v) * t`（value residual 的 lerp），最后统一过 fq_act。
    """
    va, vb, vc = to_fixed(a), to_fixed(b), to_fixed(c)
    return _finish([p + round_div(q * r, ONE) for p, q, r in zip(va, vb, vc)], quantize)


def qt_relu_sq(x, quantize=True):
    """输入：QTensor；输出：relu(x)^2。

    预期行为：对应 CMix 的 `fq_act(relu(key(k)) ** 2)`：先钳到非负再平方。
    """
    return _finish([round_div(t * t, ONE) for t in (0 if v < 0 else v for v in to_fixed(x))],
                   quantize)


def qt_sum_head(x, head_size, quantize=False):
    """输入：QTensor（长度 = head 数 * head_size）、每个 head 的长度；输出：QTensor。

    预期行为：每个 head 内求和，再把结果广播回该 head 的每个通道 —— 对应训练侧的
              `(r * k * r_k).sum(dim=-1, keepdim=True)` 再逐元素乘 `v`。
    """
    n = len(x.codes)
    assert n % head_size == 0, "长度必须是 head_size 的整数倍"
    v = to_fixed(x)
    out = []
    for base in range(0, n, head_size):
        s = sum(v[base:base + head_size])
        out.extend([s] * head_size)
    return _finish(out, quantize)
