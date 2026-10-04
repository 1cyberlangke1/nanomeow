"""定点 wkv7 递推的对照测试。

两个参照物：

1. **float 逐 token 递推**（本文件里自己写的，float64）—— 这是 wkv7 的定义式，
   定点实现必须逼近它。实测差 5e-4 量级，所以这条用紧容差。
2. **训练侧的 QAT 分块实现**（`src/wkv7.py::wkv7_chunked`）—— 松容差。
   注意：分块实现和逐 token 定义式**在 float64 下就差 2e-3 ~ 7e-2**（T 越大越大），
   这跟本文件的定点实现无关，是训练侧分块算法的数值特性；G2 闸门要正面处理它。
"""

import math
import pathlib
import sys

import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "train"))

from ref.fixed import mul, normalize, value  # noqa: E402
from ref.int8_model import QTensor  # noqa: E402
from ref.nonlinear import ONE  # noqa: E402
from ref.elemwise import qt_mul, qt_neg  # noqa: E402
from ref.wkv7 import decay_q15, normalize_p2, wkv7_recurrence  # noqa: E402
from src.qat import IntxFakeQuantizer, per_tensor_int8  # noqa: E402
from src.wkv7 import wkv7_chunked  # noqa: E402

N_HEAD, HEAD = 4, 8
C = N_HEAD * HEAD


def to_fixed_scale(scale):
    """输入：正浮点 scale；输出：(m, e)。预期行为：把 scale 落到定点表示上。"""
    mant, exp = math.frexp(scale)
    return normalize(int(round(mant * (1 << 31))), exp - 31)


def quant(tensor):
    """输入：float 张量；输出：(dequant 后的 float 张量, scale, 整数码列表)。

    预期行为：复刻训练侧的 per-tensor 动态假量化，并把码与 scale 一起交出来。
    """
    q = IntxFakeQuantizer(per_tensor_int8())
    out = q(tensor)
    s = float(q.scale)
    codes = [int(round(float(v) / s)) for v in out.reshape(-1)]
    return out, s, codes


def as_qt(codes, scale):
    """输入：整数码、scale；输出：QTensor。"""
    return QTensor(codes, to_fixed_scale(scale))


def make_inputs(seed, t_len):
    """输入：随机种子、序列长度；输出：w/q/k/v/a/b 的 (dequant, scale, codes) 三元组。"""
    torch.manual_seed(seed)
    raw = {
        "w": (torch.rand(1, t_len, C) * 2.5 + 0.5) * -1.0,   # w_in <= -0.5
        "q": torch.randn(1, t_len, C) * 1.5,
        "k": torch.randn(1, t_len, C) * 1.5,
        "v": torch.randn(1, t_len, C) * 1.5,
        "a": torch.rand(1, t_len, C),
        "b": torch.randn(1, t_len, C) * 0.5,
    }
    return {name: quant(t) for name, t in raw.items()}


def float_per_token(parts, t_len):
    """输入：make_inputs 的结果、序列长度；输出：(t_len, C) 的 float64 递推结果。

    预期行为：用**同一个 Q15 网格的 w** 做逐 token 递推（wkv7 的定义式），
              作为定点实现的紧参照；不含任何量化网格之外的近似。
    """
    w_deq = parts["w"][0].reshape(t_len, C).double()
    w15 = torch.round(torch.exp(-torch.exp(w_deq)) * 32768) / 32768
    qd, kd, vd, ad, bd = (parts[n][0].reshape(t_len, C).double() for n in ("q", "k", "v", "a", "b"))
    state = torch.zeros(N_HEAD, HEAD, HEAD, dtype=torch.float64)
    ys = []
    for t in range(t_len):
        yv = torch.zeros(C, dtype=torch.float64)
        for h in range(N_HEAD):
            sl = slice(h * HEAD, (h + 1) * HEAD)
            s = state[h]
            sa = ad[t, sl] @ s
            s = torch.diag(w15[t, sl]) @ s + torch.outer(bd[t, sl], sa) + torch.outer(kd[t, sl], vd[t, sl])
            state[h] = s
            yv[sl] = qd[t, sl] @ s
        ys.append(yv)
    return torch.stack(ys)


def qt_of(parts, name):
    """输入：make_inputs 的结果、张量名；输出：对应的 QTensor。"""
    _, scale, codes = parts[name]
    return as_qt(codes, scale)


def run_fixed(parts, t_len):
    """输入：make_inputs 的结果、序列长度；输出：(定点结果 (t_len,C), 最终 state 的 float64 张量)。"""
    state = [[[0] * HEAD for _ in range(HEAD)] for _ in range(N_HEAD)]
    y = wkv7_recurrence(
        qt_of(parts, "q"), qt_of(parts, "k"), qt_of(parts, "v"),
        qt_of(parts, "a"), qt_of(parts, "b"),
        decay_q15(qt_of(parts, "w")), state, HEAD)
    yv = torch.tensor([c * value(y.scale) for c in y.codes]).view(t_len, C).double()
    step = value(mul(qt_of(parts, "k").scale, qt_of(parts, "v").scale))
    st = torch.tensor(state, dtype=torch.float64) * step
    return yv, st


def test_qt_mul_and_neg():
    """qt_mul 的码 = 码之积；quantize=True 时必须落回 int8 码域。"""
    x = as_qt([100, -50, 7], 0.02)
    y = as_qt([3, 4, -5], 0.5)
    prod = qt_mul(x, y, quantize=False)
    assert prod.codes == [300, -200, -35]
    assert abs(value(prod.scale) - 0.01) < 1e-6
    q = qt_mul(x, y, quantize=True)
    assert max(abs(c) for c in q.codes) <= 127
    assert qt_neg(x).codes == [-100, 50, -7]


def test_normalize_p2_matches_torch():
    """normalize_p2 必须复现 fq_act(F.normalize(x.view(..., n_head, -1), dim=-1, p=2))。

    容差口径：定点路与 float32 参考在**最终 int8 码**上允许差 1 格。
    两条路的算术精度不同：定点路是「整数范数 + Q16 归一化 + 整数除法」，参考是 float32
    的 norm 与除法；实测归一化后的实值两者只差 1e-5 量级，但当元素正好落在取整边界上
    （离 .5 不到 1e-4 个码）时两边会各站一边。所以这里钉的是「没有一格差 2 以上」
    ——那才说明分组、范数或步长真算错了（分组错时实测差到 70 以上）。
    """
    torch.manual_seed(4)
    exact, total = 0, 0
    for trial in range(10):
        raw = torch.randn(1, 8, C) * (0.5 + trial * 0.5)
        fq = IntxFakeQuantizer(per_tensor_int8())
        xq = fq(raw)
        s = float(fq.scale)
        ref = fq(F.normalize(xq.view(1, 8, N_HEAD, HEAD), dim=-1, p=2.0)).reshape(-1)
        # 组数 = 展平后的 (token, head) 对数：参考的 view(1, 8, N_HEAD, HEAD) 每个 head 一组，
        # 所以是 8 个 token × N_HEAD 个头 = 32 组，组大小才是 HEAD。
        n_group = xq.numel() // HEAD
        out = normalize_p2(as_qt([int(round(float(v) / s)) for v in xq.reshape(-1)], s), n_group)
        want = [int(round(float(v) / float(fq.scale))) for v in ref]
        diffs = [abs(a - b) for a, b in zip(out.codes, want)]
        assert max(diffs) <= 1, (out.codes, want)
        exact += diffs.count(0)
        total += len(diffs)
    assert exact >= 0.99 * total, (exact, total)


def test_normalize_p2_zero_group_is_zero():
    """全零组：范数为 0，输出必须全 0（对应 F.normalize 里 max(norm, eps) 的行为）。"""
    out = normalize_p2(as_qt([0] * C, 1.0), N_HEAD)
    assert out.codes == [0] * C


def test_decay_q15_is_on_the_q15_grid():
    """decay 必须落在 Q15 网格上，且范围就是 [0.545, 1)。"""
    parts = make_inputs(0, 32)
    w15 = decay_q15(qt_of(parts, "w"))
    assert all(0 <= v <= 32768 for v in w15)
    assert all(abs(v - round(v)) == 0 for v in w15)
    lo, hi = min(w15) / 32768, max(w15) / 32768
    assert 0.5 <= lo and hi <= 1.0, (lo, hi)


def test_wkv7_matches_float_per_token():
    """定点递推必须逼近 float 逐 token 定义式：相对误差 < 2e-3。"""
    for t_len in (1, 16, 64):
        parts = make_inputs(0, t_len)
        ref = float_per_token(parts, t_len)
        got, _ = run_fixed(parts, t_len)
        rel = (got - ref).abs().max().item() / ref.abs().max().item()
        assert rel < 2e-3, (t_len, rel)


def test_wkv7_state_matches_float_per_token():
    """最终 state（换成实数单位后）也要逼近 float 逐 token：相对误差 < 2e-3。"""
    for t_len in (16, 64):
        parts = make_inputs(1, t_len)
        _, ref_y = None, float_per_token(parts, t_len)
        _, got_state = run_fixed(parts, t_len)
        ref_state = _float_state(parts, t_len)
        rel = (got_state - ref_state).abs().max().item() / ref_state.abs().max().item()
        assert rel < 2e-3, (t_len, rel, ref_y.shape)


def _float_state(parts, t_len):
    """输入：make_inputs 的结果、序列长度；输出：float64 逐 token 递推的最终 state。"""
    w_deq = parts["w"][0].reshape(t_len, C).double()
    w15 = torch.round(torch.exp(-torch.exp(w_deq)) * 32768) / 32768
    ad, bd, kd, vd = (parts[n][0].reshape(t_len, C).double() for n in ("a", "b", "k", "v"))
    state = torch.zeros(N_HEAD, HEAD, HEAD, dtype=torch.float64)
    for t in range(t_len):
        for h in range(N_HEAD):
            sl = slice(h * HEAD, (h + 1) * HEAD)
            s = state[h]
            sa = ad[t, sl] @ s
            state[h] = torch.diag(w15[t, sl]) @ s + torch.outer(bd[t, sl], sa) + torch.outer(kd[t, sl], vd[t, sl])
    return state


def test_wkv7_close_to_qat_chunked():
    """和训练侧 QAT 分块实现比：松容差（分块 vs 逐 token 在 float64 下本来就差这么多）。"""
    t_len = 32
    parts = make_inputs(0, t_len)
    raw = {n: parts[n][0].view(1, t_len, N_HEAD, HEAD) for n in ("w", "q", "k", "v", "a", "b")}
    qat = IntxFakeQuantizer(per_tensor_int8())
    ref, _ = wkv7_chunked(raw["w"], raw["q"], raw["k"], raw["v"], raw["a"], raw["b"], None, 16, qat)
    ref = ref.reshape(t_len, C).double()
    got, _ = run_fixed(parts, t_len)
    rel = (got - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 2e-2, rel
