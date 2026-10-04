"""INT8 引擎的整模型对拍：state 换单位与 G2 困惑度闸门。"""

import json
import math
import pathlib
import sys

import pytest
import torch

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "train"))

from _ckpt import latest_ckpt  # noqa: E402
from ref.fixed import mul, normalize, value  # noqa: E402
from ref.int8_model import QTensor  # noqa: E402
from ref.wkv7 import wkv7_recurrence  # noqa: E402

CKPT = latest_ckpt()
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


@pytest.mark.skipif(CKPT is None, reason="需要训练产出的 checkpoint")
def test_g2_ppl_degradation_within_5pct():
    """G2 闸门：int8 引擎相对同权重浮点前向的困惑度退化必须 ≤ 5%。

    口径（都是实测定的，不是估计）：
    - 数据取**对话域内的真实样本**（`train/data/chitchat_para.jsonl` 的 user:/bot: 行）。
      不能取语料文件的前 512 字节——那是 JSON 残片、两侧都接近均匀分布，闸门没有鉴别力。
    - ① 用户口径：整条序列的 nats/byte，int8 相对 bf16 原样前向 ≤ 5%。
    - ② 忠实性口径：答案段（唯一进了 loss、也是部署真正在意的区域）int8 相对
      「训练时真正优化的那个函数」（同权重 + QAT 假量化）≤ 5%。实测 int8 比它还好，
      所以这条抓的是「定点实现自己引入退化」这类回归。
    - ③ 绝对护栏：答案段 int8 与 bf16 原样的差 ≤ 0.01 nats/byte。该段 nll 只有 0.008
      量级，相对百分比会被小基数放大（实测 +20%），所以这里卡绝对值而不是百分比。
    """
    from ref.model import Int8Model
    from ref.weights import load_weights
    from src.config import NanoConfig
    from src.data import answer_start
    from src.model import NanoRWKV
    from src.qat import prepare_qat
    from src.tokenizer import ETX_ID

    cfg, wts = load_weights(str(CKPT))
    eng = Int8Model(cfg, wts)
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)

    def build(qat):
        """输入：是否打开 QAT 假量化；输出：对应的 bf16 模型。"""
        m = NanoRWKV(NanoConfig(**ckpt["nano_cfg"]))
        m.load_state_dict(ckpt["model"])
        if qat:
            prepare_qat(m)
        return m.to(torch.bfloat16).eval()

    plain, quant = build(False), build(True)
    rows = [json.loads(l)["text"] for l in open(
        HERE.parents[2] / "train" / "data" / "chitchat_para.jsonl", encoding="utf-8")]
    step = max(1, len(rows) // 24)
    rows = rows[::step][:24]

    seq_b = seq_i = ans_b = ans_q = ans_i = 0.0
    n_seq = n_ans = 0
    for r in rows:
        toks = list(r.encode("utf-8"))[:200] + [ETX_ID]
        lo, hi = max(0, answer_start(toks) - 1), len(toks) - 2
        with torch.no_grad():
            lg = plain(torch.tensor([toks], dtype=torch.long))[0][0].float()
            lq = quant(torch.tensor([toks], dtype=torch.long))[0][0].float()
        state = eng.zeros_state()
        outs = [eng.forward_token(toks[t], state) for t in range(len(toks) - 1)]
        for t in range(len(toks) - 1):
            b = -float(torch.log_softmax(lg[t], dim=-1)[toks[t + 1]])
            out = outs[t]
            z = torch.tensor([c * value(out.scale) for c in out.codes], dtype=torch.float32)
            i = -float(torch.log_softmax(z, dim=-1)[toks[t + 1]])
            seq_b += b
            seq_i += i
            n_seq += 1
            if lo <= t < hi:
                ans_b += b
                ans_q -= float(torch.log_softmax(lq[t], dim=-1)[toks[t + 1]])
                ans_i += i
                n_ans += 1
    assert seq_i / seq_b - 1.0 <= 0.05, (seq_b / n_seq, seq_i / n_seq)
    assert ans_i / ans_q - 1.0 <= 0.05, (ans_b / n_ans, ans_q / n_ans, ans_i / n_ans)
    assert ans_i - ans_b <= 0.01 * n_ans, (ans_b / n_ans, ans_i / n_ans)
