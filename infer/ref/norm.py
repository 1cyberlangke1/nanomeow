"""定点归一化：LayerNorm 与 GroupNorm 共用一套实现，只差组数。

训练侧口径（照 `train/src/model.py` 复刻，eps 一个都不许改）：

    Block._norm : F.layer_norm(x, (C,), fq(w), fq(b), 1e-5)        # nn.LayerNorm 默认 eps
    Tmix        : F.group_norm(x.view(B*T, C), n_head, fq(w), fq(b), 64e-5)
    NanoRWKV    : F.layer_norm(x, (C,), fq(w), fq(b), 1e-5)

两者都是「按 groups 个一组，组内做 (x - mean) / sqrt(var + eps)，再逐通道仿射」，
所以这里只有一套实现：LayerNorm 就是 groups = 1。

定点口径：

- 激活先落到 Q16（`to_fixed`），均值是 Q16、方差是 Q32（偏差 d 是 Q16，d^2 就是 Q32）；
- 方差是**有偏**的（除以组内元素数），和 PyTorch 的 LayerNorm / GroupNorm 一致；
- **eps 必须用 Q32 表示**：1e-5 在 Q16 下只有 0.65，直接取整是 50% 的相对误差，
  所以把 eps 下移 16 位到 Q32 再和方差相加；
- rsqrt 用「整数平方根 + 一次除法」：先把被开方数左移 2 * RSQRT_EXTRA 位再开方，
  开方的截断误差相对量级降到 2^-26 上下，比直接在 Q32 上开方精确得多。
"""

from .fixed import round_div
from .nonlinear import FRAC_BITS, ONE, from_fixed, to_fixed

RSQRT_EXTRA = 10        # rsqrt 前把被开方数左移的位数（2 * RSQRT_EXTRA）


def isqrt(n):
    """输入：非负整数 n；输出：floor(sqrt(n))。

    预期行为：牛顿法，初值取 2^ceil(bit_length/2)（一定不小于真实根），
              收敛到 floor(sqrt(n)) 后停下。
    """
    if n <= 0:
        return 0
    x = 1 << ((n.bit_length() + 1) // 2)
    while True:
        y = (x + n // x) >> 1
        if y >= x:
            return x
        x = y


def rsqrt_q32(x):
    """输入：Q32 的正整数 x；输出：Q16 的 1 / sqrt(x / 2^32)。

    预期行为：1 / sqrt(x / 2^32) = 2^16 / sqrt(x)，落到 Q16 就是 2^32 / sqrt(x)。
              先把 x 左移 2 * RSQRT_EXTRA 位再开方（分子补回 2^RSQRT_EXTRA），
              这样开方结果的截断误差相对量级只有 2^-(16 + RSQRT_EXTRA)。
              最后一次除法带舍入：结果是 Q16，1/sqrt(x) 很小时它的绝对分辨率就是
              一个量子（例：1/sqrt(10000) = 0.01 在 Q16 下只有 655 格），
              不舍入会白丢半格。
              x <= 0 返回 0（调用方保证已经加过 eps）。
    """
    if x <= 0:
        return 0
    s = isqrt(x << (2 * RSQRT_EXTRA))
    return round_div(1 << (2 * FRAC_BITS + RSQRT_EXTRA), s)


def norm_q(x, weight, bias, eps, groups=1):
    """输入：QTensor x（长度 = groups * 组大小）、QTensor weight / bias（同长）、eps、组数；
    输出：QTensor。

    预期行为：逐组算均值与有偏方差，做 (v - mean) * rsqrt(var + eps)，再逐通道乘 weight 加 bias，
              最后按 per-tensor 动态口径量化回 int8 —— 对应训练侧的 `fq_act(F.layer_norm(...))`。
              weight / bias 的 scale 各按自己的 QTensor 走，不假设和 x 同 scale。
    """
    n = len(x.codes)
    assert len(weight.codes) == n and len(bias.codes) == n, "weight / bias 长度必须与 x 相同"
    assert groups >= 1 and n % groups == 0, "长度必须能被组数整除"
    size = n // groups
    v = to_fixed(x)
    w = to_fixed(weight)
    b = to_fixed(bias)
    eps_q32 = int(round(eps * (1 << (2 * FRAC_BITS))))
    out = []
    for gi in range(groups):
        base = gi * size
        group = v[base:base + size]
        mean = round_div(sum(group), size)
        acc = 0
        for val in group:
            d = val - mean
            acc += d * d
        var_q32 = round_div(acc, size)
        r = rsqrt_q32(var_q32 + eps_q32)
        for i in range(size):
            k = base + i
            normed = round_div((v[k] - mean) * r, ONE)
            out.append(round_div(normed * w[k], ONE) + b[k])
    return from_fixed(out)
