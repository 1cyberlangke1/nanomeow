"""纯整数前向：量化张量、GEMV 与动态再量化。

输入：QTensor（整数码 + 一个 scale）与 QMatrix（int8 权重 + 每行一个 scale）。
输出：QTensor。
预期行为：只用整数算术复刻训练侧的 QAT 假量化——激活 per-tensor 动态、权重 per-row，
          矩阵乘之后再过一次 per-tensor 动态假量化。C 引擎必须逐位复刻本文件（G1 闸门）。
"""

from .fixed import (
    MANT_MIN,
    QMAX,
    div_int,
    mul,
    mul_int,
    normalize,
    quantize_dynamic,
    round_div,
)

INT8_MIN = -128
INT8_MAX = 127

# 行 scale 拉到公共指数时允许的最大左移。超过这个宽度，行之间的实值差已经超出
# 32/64 位定点能表达的范围，低位行的贡献会被舍掉——这是定点宽度的物理限制，不是 bug。
MAX_ALIGN_SHIFT = 40


class QTensor:
    """量化张量：整数码 + 一个 scale（所有元素共用）。"""

    __slots__ = ("codes", "scale")

    def __init__(self, codes, scale):
        """输入：码列表（整数）、scale；输出：量化张量。"""
        self.codes = list(codes)
        self.scale = scale

    def __len__(self):
        """输入：无；输出：元素个数。"""
        return len(self.codes)

    def __repr__(self):
        """输入：无；输出：可读字符串。"""
        return "QTensor(n=%d, scale=%s)" % (len(self.codes), self.scale)


class QMatrix:
    """量化矩阵：int8 码（行主序）+ 每行一个 scale。"""

    __slots__ = ("codes", "scales", "rows", "cols")

    def __init__(self, codes, scales, rows, cols):
        """输入：码列表、每行 scale 列表、行数、列数；输出：量化矩阵。"""
        assert len(codes) == rows * cols and len(scales) == rows
        self.codes = list(codes)
        self.scales = list(scales)
        self.rows = rows
        self.cols = cols

    def __repr__(self):
        """输入：无；输出：可读字符串。"""
        return "QMatrix(%dx%d)" % (self.rows, self.cols)


def quantize(codes, scale):
    """输入：整数码、scale；输出：QTensor。预期行为：per-tensor 动态假量化回 int8。"""
    q, s = quantize_dynamic(codes, scale)
    return QTensor(q, s)


def _quantize_rows(accs, scales):
    """输入：每行一个整数码、每行一个 scale（= s_w[i] * s_x）；输出：统一 scale 的 QTensor。

    预期行为：复刻训练侧对 Linear 输出的 per-tensor 动态假量化（SYMMETRIC_NO_CLIPPING_ERR）。
              每行实值是 acc_i * scale_i，所以先把各行拉到公共指数（取最小指数）上比较，
              再取步长并量化；步长写成有理数 num / den，全程整数。
              注意 scale_i 已经含了输入的 s_x，这里**不能**再乘一次。
    """
    e_ref = min(e for _, e in scales)
    vals = []
    for acc, (m, e) in zip(accs, scales):
        vals.append(acc * m * (1 << min(e - e_ref, MAX_ALIGN_SHIFT)))
    mx, mn = max(vals), min(vals)
    num = max(mx * (QMAX + 1), -mn * QMAX)
    den = QMAX * (QMAX + 1)
    base = normalize(1, e_ref)
    if num == 0:
        return QTensor([0] * len(vals), base)
    out_scale = div_int(mul_int(base, num), den)
    out = []
    for v in vals:
        q = round_div(v * den, num)
        out.append(INT8_MIN if q < INT8_MIN else (INT8_MAX if q > INT8_MAX else q))
    return QTensor(out, out_scale)


def linear(x, mat):
    """输入：QTensor x（长度 = mat.cols）、QMatrix；输出：QTensor。

    预期行为：acc_i = sum_j w_ij * x_j，实值 = acc_i * s_w[i] * s_x；再按 per-tensor
              动态口径量化回 int8，对应训练侧的 fq_act(F.linear(fq_act(x), fq_w(W)))。

    注意训练侧只量化权重、**不再单独量化输入激活**这一处：模型里输入已经在上一个
    fq_act 点被量化过，所以这里直接用 x 的码与 scale，不重复量化。
    """
    assert len(x.codes) == mat.cols, "输入宽度与权重列数不一致"
    accs, scales = [], []
    for i in range(mat.rows):
        base = i * mat.cols
        acc = 0
        for j in range(mat.cols):
            acc += mat.codes[base + j] * x.codes[j]
        accs.append(acc)
        scales.append(mul(mat.scales[i], x.scale))
    return _quantize_rows(accs, scales)
