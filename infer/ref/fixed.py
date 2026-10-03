"""纯整数定点工具：scale 的表示、算术与动态量化。

scale 的表示与导出的权重 scale 同格式：value = m * 2 ** e，m 归一化到 [2^30, 2^31)。
所有函数只用整数运算，除法一律「四舍六入五成双」（与训练侧 torch.round 一致）。

为什么要自己算 scale 比值：训练侧 QAT 的激活量化是 **per-tensor 动态**（每次前向按当前张量
重算 max），所以 requant 乘子不能像 CMSIS-NN 那样离线烧死，必须在板子上用整数算出来。
本模块就是那套整数算式的唯一实现，C 引擎必须逐位复刻它（G1 闸门）。
"""

MANT_BITS = 30
MANT_MIN = 1 << MANT_BITS        # 归一化下界：m ∈ [2^30, 2^31)
MANT_MAX = (1 << 31) - 1
QMAX = 127                       # 对称 int8 的正端边界（负端是 -128）


def round_div(num, den):
    """输入：整数分子 num、正数分母 den；输出：round(num / den)，四舍六入五成双。

    预期行为：与 Python 的 round() / torch.round() 的中点位取偶一致；den 必须 > 0。
    """
    assert den > 0
    q, r = divmod(num, den)
    twice = 2 * r
    if twice > den or (twice == den and (q & 1)):
        q += 1
    return q


def normalize(m, e):
    """输入：任意正整数乘子 m、指数 e；输出：归一化后的 (m, e)，m ∈ [2^30, 2^31)。

    预期行为：只挪动指数，值 m * 2 ** e 不变；m == 0 时原样返回 (0, 0)。
    """
    if m == 0:
        return 0, 0
    while m < MANT_MIN:
        m <<= 1
        e -= 1
    while m > MANT_MAX:
        m >>= 1
        e += 1
    return m, e


def value(scale):
    """输入：(m, e)；输出：浮点值，仅用于测试与打印，引擎里不许出现。"""
    return scale[0] * (2.0 ** scale[1])


def mul(a, b):
    """输入：两个 scale；输出：乘积 scale（值 = value(a) * value(b)）。

    预期行为：乘子先按 2^30 缩放，指数相应加 30 再由 normalize 归一化。
              漏掉这个 30 会让结果整整差 2^30——注意「乘完再除回来」的往返测试
              抓不到这个错，必须直接对值。
    """
    return normalize(a[0] * b[0], a[1] + b[1])


def div(a, b):
    """输入：两个 scale；输出：商 scale（值 = value(a) / value(b)）。"""
    return normalize((a[0] << MANT_BITS) // b[0], a[1] - b[1] - MANT_BITS)


def apply_scale(val, s):
    """输入：整数 val、scale s；输出：round(val * value(s))。

    预期行为：把「值乘上 scale」这个动作也做成纯整数；e 为负时用一次带舍入的右移。
    """
    m, e = s
    if e >= 0:
        return val * m << e
    return round_div(val * m, 1 << (-e))


def mul_int(a, k):
    """输入：scale、正整数 k；输出：scale * k。"""
    return normalize(a[0] * k, a[1])


def div_int(a, k, bits=32):
    """输入：scale、正整数 k；输出：scale / k（乘子按四舍六入五成双）。

    预期行为：先把乘子左移 bits 位再除，否则商的低位会在归一化时被整段丢掉
              （实测只左移 0 位时相对误差约 3e-6，移到 32 位后回到 ~5e-10）。
              乘子 < 2^31、bits = 32 时中间值 < 2^63，C 里一次 64/32 位除法即可。
    """
    return normalize(round_div(a[0] << bits, k), a[1] - bits)


def ratio(a, b):
    """输入：两个 scale；输出：比值 scale（= div(a, b)）。

    预期行为：requant 的乘子就是它的乘子、移位就是它的指数，读起来比 div 更贴场景。
    """
    return div(a, b)


def quantize_dynamic(codes, scale):
    """输入：整数码列表、它们的 scale；输出：(int8 码列表, 新 scale)。

    预期行为：复刻训练侧 per-tensor 动态假量化，映射口径是 SYMMETRIC_NO_CLIPPING_ERR——
              正端除 127、负端除 128，取大者作为步长 step；
              q = clamp(round(code / step), -128, 127)，新 scale = 旧 scale * step。
              步长写成有理数 num / den（num = max(max*128, |min|*127)，den = 127*128），
              所以整个过程只有整数乘除，没有浮点。全零张量原样返回。
    """
    mx = max((int(c) for c in codes), default=0)
    mn = min((int(c) for c in codes), default=0)
    num = max(mx * (QMAX + 1), -mn * QMAX)
    den = QMAX * (QMAX + 1)
    if num == 0:
        return [0] * len(codes), scale
    new_scale = div_int(mul_int(scale, num), den)
    out = []
    for c in codes:
        q = round_div(int(c) * den, num)
        out.append(-128 if q < -128 else (127 if q > 127 else q))
    return out, new_scale
def rescale(val, scale, frac_bits):
    """输入：整数码 val（实值 = val * value(scale)）、目标小数位 frac_bits；
    输出：Q(frac_bits) 整数 = round(val * value(scale) * 2^frac_bits)。

    预期行为：把 scale 的指数整体加上 frac_bits，一次乘加移位就落到目标定点上；
              指数为负时走一次带舍入的右移。C 引擎里就是 64 位乘 + 移位 / 带舍入除法。
    """
    m, e = scale
    shift = e + frac_bits
    if shift >= 0:
        return val * m << shift
    return round_div(val * m, 1 << (-shift))


def scale_from_frac(frac_bits):
    """输入：小数位数 frac_bits；输出：值 = 2 ** -frac_bits 的 scale。

    预期行为：定点整数 v 乘上这个 scale 就还原成实值 v / 2^frac_bits。
    """
    return normalize(1, -frac_bits)
