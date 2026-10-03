"""定点非线性：exp / log1p 两张 LUT，以及 sigmoid / tanh / softplus。

为什么两张表就够：模型里所有非线性都能化成「非正数上的 exp」与「log(1 + t), t ∈ [0, 1]」两步：

    sigmoid(x)  = 1 / (1 + exp(-x))     x >= 0 走 exp(-x)；x < 0 走 exp(x) / (1 + exp(x))
    tanh(x)     = 2 * sigmoid(2x) - 1
    softplus(x) = log(1 + exp(x))       x <= 0 直接用；x > 0 拆成 x + log(1 + exp(-x))

两条支路都只让 exp 吃非正数，所以「非正数 exp」一张表就覆盖了全部调用点。

**exp：减 ln2 归约 + 257 项残差表。**
把 x 拆成 x = n * ln2 + r（n = floor(x / ln2) <= 0，r ∈ [0, ln2)），
exp(x) = 2^n * exp(r)，exp(r) ∈ [1, 2) 由 257 项表线性插值。
表项存的是 **(exp(r) - 1) 的 Q15**（∈ [0, 32768]），这样正好塞进 uint16_t ——
直接存 exp(r) 会到 65536，Q15 装不下。插值相对误差约 1e-6（步长 ln2/256 的 h^2/8），
表量化约 1.5e-5，定点位置常数约 2.4e-5，合计约 3e-5 相对 ——
比「一张 [-16, 0] 大表」的 4.9e-4 好一个数量级还多。
n < -17 时结果 < 2^-17，Q16 下舍入成 0，等于向零饱和。

**log1p：257 项 Q15 表，覆盖 t ∈ [0, 1]**（softplus 里 1 + exp(·) 的实值必落在这个区间）。
插值绝对误差约 2e-6，表量化约 1.5e-5。

两张表的步长都取 2 的幂：exp 残差表是 1/256、log1p 也是 1/256，查表位置统一是 Q8，
所以索引是右移 8、插值权重是低 8 位掩码，整条路没有除法。
exp 的归约除法写成「非负除法再取负」，C 里就是一次无符号除法，
不依赖负数除法是向零截断还是向下取整。
"""

import math

from .fixed import (
    quantize_dynamic,
    rescale,
    round_div,
    scale_from_frac,
)
from .int8_model import QTensor

FRAC_BITS = 16                  # 内部定点的小数位
ONE = 1 << FRAC_BITS            # 定点里的 1.0

EXP_LUT_Q = 15                  # 两张表都是 Q15
LUT_STEP_SHIFT = 8              # 两张表的查表位置都是 Q8（步长 1/256）
EXP_LUT_STEP_SHIFT = LUT_STEP_SHIFT
LOG1P_LUT_STEP_SHIFT = LUT_STEP_SHIFT

LN2_Q16 = int(round(math.log(2.0) * ONE))          # ln2 的 Q16 表示


def _build_exp_lut():
    """输入：无；输出：257 项 Q15 表，第 i 项 = round((2^(i/256) - 1) * 2^15)。

    预期行为：i = 0 是 0、i = 256 是 2^15（即 exp(ln2) - 1 = 1），两端都在 uint16 内。
    """
    return [int(round((2.0 ** (i / 256.0) - 1.0) * (1 << EXP_LUT_Q)))
            for i in range(257)]


def _build_log1p_lut():
    """输入：无；输出：257 项 Q15 表，第 i 项 = round(log(1 + i/256) * 2^15)。"""
    return [int(round(math.log1p(i / 256.0) * (1 << EXP_LUT_Q)))
            for i in range(257)]


EXP_LUT = _build_exp_lut()
LOG1P_LUT = _build_log1p_lut()


def _lerp_q15(lut, pos, step_shift):
    """输入：Q15 表、Q(step_shift) 的查表位置、步长指数；输出：Q(FRAC_BITS) 的线性插值。

    预期行为：低位 step_shift 位是插值权重、高位是表索引；Q15 表项先左移到 Q(FRAC_BITS)。
              位置先钳到表的右端点之内（插值位置算出来的舍入可能正好顶到 len(lut)-1），
              这样索引最大只到 len(lut)-2，hi 一定存在。
    """
    max_pos = ((len(lut) - 1) << step_shift) - 1
    if pos > max_pos:
        pos = max_pos
    idx = pos >> step_shift
    frac = pos & ((1 << step_shift) - 1)
    lo = lut[idx]
    hi = lut[idx + 1]
    out = lo << (FRAC_BITS - EXP_LUT_Q)
    return out + round_div((hi - lo) * frac, 1 << (EXP_LUT_Q + step_shift - FRAC_BITS))


def _floor_div_ln2(x):
    """输入：Q(FRAC_BITS) 整数 x <= 0；输出：floor(x / ln2)（非正）。

    预期行为：写成「非负除法再取负」，C 里就是一次无符号除法，
              不依赖负数除法是向零截断还是向下取整。
    """
    return -(((-x) + LN2_Q16 - 1) // LN2_Q16)


def exp_q(x):
    """输入：Q(FRAC_BITS) 整数 x（调用方保证 x <= 0）；输出：Q(FRAC_BITS) 的 exp(x) ∈ (0, 1]。

    预期行为：x >= 0 返回 1；x < -17 * ln2 直接返回 0；其余走 ln2 归约 + 残差表插值。
    """
    if x >= 0:
        return ONE
    n = _floor_div_ln2(x)
    if n < -FRAC_BITS - 1:
        return 0
    r = x - n * LN2_Q16
    # 查表位置 pos = (r / ln2) * 256，取 Q8：pos = r * 65536 / LN2_Q16。
    # 这里必须**除法**而不是乘一个定点常数：常数取整会带来系统性正偏
    # （实测 3.5e-5 相对，让 25% 的 decay 码比参考高 1 格），除法只差半个最低位。
    # r <= LN2_Q16 < 2^16，所以 r << 16 < 2^32，C 里一次 uint32 加 + 32 位除法就够，
    # 不需要 64 位除法。
    pos = round_div(r << 16, LN2_Q16)
    value = ONE + _lerp_q15(EXP_LUT, pos, EXP_LUT_STEP_SHIFT)
    if n == 0:
        return value
    return round_div(value, 1 << (-n))


def log1p_q(t):
    """输入：Q(FRAC_BITS) 整数 t ∈ [0, 1]；输出：Q(FRAC_BITS) 的 log(1 + t) ∈ [0, ln2]。"""
    if t <= 0:
        return 0
    if t > ONE:
        t = ONE
    return _lerp_q15(LOG1P_LUT, t, LOG1P_LUT_STEP_SHIFT)


def sigmoid_q(x):
    """输入：Q(FRAC_BITS) 整数；输出：Q(FRAC_BITS) 的 sigmoid(x) ∈ (0, 1)。

    预期行为：x >= 0 走 1 / (1 + exp(-x))、x < 0 走 exp(x) / (1 + exp(x))；
              两条支路的 exp 都吃非正数，不会跑到表定义域外。分子要 64 位（2^32）。
    """
    if x >= 0:
        e = exp_q(-x)
        return round_div(ONE << FRAC_BITS, ONE + e)
    e = exp_q(x)
    return round_div(e << FRAC_BITS, ONE + e)


def tanh_q(x):
    """输入：Q(FRAC_BITS) 整数；输出：Q(FRAC_BITS) 的 tanh(x) ∈ (-1, 1)。

    预期行为：|x| > 16 时先夹住（tanh 早就饱和了，夹住也避免 2 * x 溢出）。
    """
    limit = 16 * ONE
    if x > limit:
        x = limit
    elif x < -limit:
        x = -limit
    return 2 * sigmoid_q(2 * x) - ONE


def softplus_q(x):
    """输入：Q(FRAC_BITS) 整数；输出：Q(FRAC_BITS) 的 softplus(x) = log(1 + exp(x)) >= 0。

    预期行为：x <= 0 走 log1p(exp(x))；x > 0 拆成 x + log1p(exp(-x))，exp 仍只吃非正数；
              两侧 16 之外饱和（x 很大时 softplus(x) 就是 x）。
    """
    limit = 16 * ONE
    if x >= limit:
        return x
    if x <= -limit:
        return 0
    if x <= 0:
        return log1p_q(exp_q(x))
    return x + log1p_q(exp_q(-x))


def to_fixed(x):
    """输入：QTensor；输出：Q(FRAC_BITS) 整数列表（实值 = 整数 / 2^FRAC_BITS）。"""
    return [rescale(c, x.scale, FRAC_BITS) for c in x.codes]


def from_fixed(vals):
    """输入：Q(FRAC_BITS) 整数列表；输出：QTensor。

    预期行为：复刻训练侧的 `fq_act(非线性(...))` —— 非线性输出先落回实数
              （scale = 2^-FRAC_BITS），再按 per-tensor 动态口径量化回 int8。
    """
    q, s = quantize_dynamic(vals, scale_from_frac(FRAC_BITS))
    return QTensor(q, s)


def qt_exp(x):
    """输入：QTensor（实值必须 <= 0）；输出：QTensor。预期行为：逐元素 exp 后再量化。"""
    return from_fixed([exp_q(v) for v in to_fixed(x)])


def qt_sigmoid(x):
    """输入：QTensor；输出：QTensor。预期行为：逐元素 sigmoid 后再 per-tensor 量化。"""
    return from_fixed([sigmoid_q(v) for v in to_fixed(x)])


def qt_tanh(x):
    """输入：QTensor；输出：QTensor。预期行为：逐元素 tanh 后再 per-tensor 量化。"""
    return from_fixed([tanh_q(v) for v in to_fixed(x)])


def qt_softplus(x):
    """输入：QTensor；输出：QTensor。预期行为：逐元素 softplus 后再 per-tensor 量化。"""
    return from_fixed([softplus_q(v) for v in to_fixed(x)])
