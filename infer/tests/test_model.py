"""INT8 引擎的整模型对拍：state 换单位与 G2 困惑度闸门。"""

import math
import pathlib
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "train"))

from ref.fixed import mul, normalize, value  # noqa: E402
from ref.int8_model import QTensor  # noqa: E402
from ref.wkv7 import wkv7_recurrence  # noqa: E402

CKPT = HERE.parents[2] / "train" / "out" / "sft_dyn_fine" / "sft_step2750.pth"
N = 2                      # 测试用的 head_size
Q7 = normalize(1, -7)      # 值 = 2^-7 的 scale（激活 / 向量）
Q9 = normalize(1, -9)      # 第一趟的 k / v：step = 2^-18
Q7V = normalize(1, -7)     # 第二趟的 k，配 Q8 的 v：step = 2^-15


def test_recurrence_rescales_state_to_new_step():
    """state 换单位：第二趟 step 与第一趟不同时，旧 state 必须先换算再递推。

    预期行为：引擎两趟递推后的 state，与 fp64 复算的
              `round((w * S + bᵀ(a @ S) + kᵀv) / step) * step` 相差 ≤ 1.5 个 step；
              不传 step_prev（旧口径，直接拿旧整数按新单位读）时误差远大于此。
    """
    k1, v1 = QTensor([100, -60], Q9), QTensor([80, 120], normalize(1, -9))
    a1, b1 = QTensor([64, -32], Q7), QTensor([32, 64], Q7)
    k2, v2 = QTensor([90, -70], Q7V), QTensor([110, 40], normalize(1, -8))
    a2, b2 = QTensor([48, 16], Q7), QTensor([24, 48], Q7)
    w1, w2 = [32768, 16384], [16384, 32768]      # Q15 衰减：1.0 / 0.5

    def run(step_prev):
        """输入：传给第二趟的 step_prev（None = 旧口径）；输出：两趟后的 state。"""
        state = [[[0] * N for _ in range(N)]]
        wkv7_recurrence(QTensor([10, 20], Q7), k1, v1, a1, b1, w1, state, N, None)
        wkv7_recurrence(QTensor([30, 10], Q7), k2, v2, a2, b2, w2, state, N, step_prev)
        return state[0]

    def reals(mat, scale):
        """输入：整数矩阵、它的 scale；输出：实值矩阵（fp64）。"""
        return torch.tensor(mat, dtype=torch.float64) * value(scale)

    step1, step2 = mul(k1.scale, v1.scale), mul(k2.scale, v2.scale)
    # 第一趟从零状态起步，state 就是 kᵀv（step1 的整数倍，无额外取整）
    state1 = [[[0] * N for _ in range(N)]]
    wkv7_recurrence(QTensor([10, 20], Q7), k1, v1, a1, b1, w1, state1, N, None)
    s1 = reals(state1[0], step1)
    a2r = torch.tensor([c * value(a2.scale) for c in a2.codes], dtype=torch.float64)
    b2r = torch.tensor([c * value(b2.scale) for c in b2.codes], dtype=torch.float64)
    k2r = torch.tensor([c * value(k2.scale) for c in k2.codes], dtype=torch.float64)
    v2r = torch.tensor([c * value(v2.scale) for c in v2.codes], dtype=torch.float64)
    w2r = torch.tensor(w2, dtype=torch.float64) / 32768.0
    ref = (w2r.unsqueeze(-1) * s1
           + torch.outer(b2r, a2r @ s1) + torch.outer(k2r, v2r))
    ref = torch.round(ref / value(step2)) * value(step2)

    good = reals(run(step1), step2)
    bad = reals(run(None), step2)
    assert float((good - ref).abs().max()) / value(step2) <= 1.5, (good - ref).abs().max() / value(step2)
    assert float((bad - ref).abs().max()) / value(step2) > 5.0, (bad - ref).abs().max() / value(step2)


@pytest.mark.skipif(not CKPT.exists(), reason="需要训练产出的 checkpoint")
def test_g2_ppl_degradation_within_5pct():
    """G2 闸门：int8 引擎与同权重 bf16 前向的困惑度退化必须 ≤ 5%。

    预期行为：同一段字节上，引擎逐 token 前向的 nats/byte 相对 bf16 全序列前向
              不超过 5%；两侧 logits 的 argmax 允许有差异（量化误差），所以只卡困惑度。
    """
    from ref.model import Int8Model
    from ref.weights import load_weights
    from src.config import NanoConfig
    from src.model import NanoRWKV

    cfg, wts = load_weights(str(CKPT))
    eng = Int8Model(cfg, wts)
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    plain = NanoRWKV(NanoConfig(**ckpt["nano_cfg"]))
    plain.load_state_dict(ckpt["model"])
    plain = plain.to(torch.bfloat16).eval()

    data = (HERE.parents[2] / "train" / "data" / "chitchat_seed.jsonl").read_text(
        encoding="utf-8").encode("utf-8")[:512]
    toks = list(data)

    with torch.no_grad():
        lg = plain(torch.tensor([toks], dtype=torch.long))[0][0].float()
    tot_b = 0.0
    for t in range(len(toks) - 1):
        tot_b -= float(torch.log_softmax(lg[t], dim=-1)[toks[t + 1]])
    nll_b = tot_b / (len(toks) - 1)

    st = eng.zeros_state()
    tot_i = 0.0
    for t in range(len(toks) - 1):
        out = eng.forward_token(toks[t], st)
        z = torch.tensor([c * value(out.scale) for c in out.codes], dtype=torch.float32)
        tot_i -= float(torch.log_softmax(z, dim=-1)[toks[t + 1]])
    nll_i = tot_i / (len(toks) - 1)

    assert nll_i / nll_b - 1.0 <= 0.05, (nll_b, nll_i, math.exp(nll_b), math.exp(nll_i))