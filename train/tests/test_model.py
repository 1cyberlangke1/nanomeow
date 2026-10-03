"""模型层测试：参数量、初始化、因果性、状态续接、全参数可训练、单批过拟合。"""

import math

import pytest
import torch

from src.config import NanoConfig
from src.model import DEAD_BY_REFERENCE_BRANCH, NanoRWKV, RWKVState

TINY = NanoConfig()


def _batch(batch=2, ctx=16, vocab=8, seed=0):
    """输入：batch/ctx/vocab/seed；输出：(idx (B,T) int64, y (B,T) int64) 的随机样本。"""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, vocab, (batch, ctx + 1), generator=g)
    return idx[:, :-1].contiguous(), idx[:, 1:].contiguous()


def _loss(logits, y, mask=None):
    """输入：logits (B,T,V)、y (B,T)、可选 mask (B,T)；输出：标量 loss。"""
    flat = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none"
    ).view(y.shape)
    if mask is None:
        return flat.mean()
    return (flat * mask).sum() / mask.sum().clamp_min(1)


def test_param_count_matches_formula():
    """实测参数量必须等于 config 的解析公式。"""
    model = NanoRWKV(TINY)
    assert model.parameter_count() == TINY.param_count()


def test_head_is_independent_of_emb():
    """head 必须是独立矩阵，不许与 emb 共享权重（原版行为）。"""
    model = NanoRWKV(TINY)
    assert model.head.weight is not model.emb.weight
    names = [n for n, _ in model.named_parameters()]
    assert "head.weight" in names and "emb.weight" in names


def test_initial_loss_matches_logit_variance():
    """初始 loss 必须等于 ln(V) + Var(logits)/2（小方差下的高斯交叉熵恒等式）。

    不能直接断言 loss ≈ ln(V)：参考的 head 初始化是正交 gain = 0.5*sqrt(V/C)，
    logits 不是 0 而是有方差（实测 ~0.24），loss 因此比 ln(V) 高一点点。
    恒等式才是真正的规格，常数只是它的一个特例。
    """
    model = NanoRWKV(NanoConfig())
    idx, y = _batch(batch=8, ctx=64, vocab=256, seed=7)
    with torch.no_grad():
        logits, _ = model(idx)
    loss = _loss(logits, y).item()
    expected = math.log(256) + logits.var().item() / 2
    assert abs(loss - expected) < 0.08, f"loss={loss:.4f} 期望≈{expected:.4f}"
    assert logits.var().item() < 1.0, "初始 logits 方差过大，说明 head 初始化不对"


def test_all_params_receive_nonzero_grad():
    """每个参数都必须进计算图，且训练起来之后梯度非零（白名单为空）。

    分两步查，因为第 0 步的零梯度是官方初始化的**预期结果**、不是漏训：
    `att.output.weight` 与 `ffn.value.weight` 被官方初始化成 0，于是第 0 步
    反向时「上游到 attention/ffn 内部的梯度」恒为 0（梯度只从残差旁路流过）。
    这层零初始化一被 AdamW 推开，梯度就通了。
    """
    model = NanoRWKV(TINY)
    model.train()
    idx, y = _batch()

    # 第一步：所有参数都要挂在计算图上（grad 非 None）。唯一的例外是
    # 显式登记的第 0 层 v0/v1/v2——参考的按层分支让它们结构性死掉。
    logits, _ = model(idx)
    _loss(logits, y).backward()
    actually_missing = {n for n, p in model.named_parameters() if p.grad is None}
    assert actually_missing == DEAD_BY_REFERENCE_BRANCH, (
        f"死参数集合与登记不符：{sorted(actually_missing)} != {sorted(DEAD_BY_REFERENCE_BRANCH)}"
    )
    missing = [n for n, p in model.named_parameters()
               if p.grad is None and n not in DEAD_BY_REFERENCE_BRANCH]
    assert missing == [], f"这些参数没进计算图：{missing}"

    # 第二步：走几步让零初始化层被推开，再查「梯度恒为零」的死参数。
    # 20 步而不是 5 步：参考把 `.att.output.` / `ffn.value.` / `w1,a1,v1,g1` 全初始化成 0，
    # 这些零块要靠旁路梯度一步步「解开」（实测 fp32 约 3~4 步；QAT 下中间量被量化压到 0、
    # 逃逸慢约一倍，约 8 步）。窗口太短会把「还没解开」误判成「漏训」。
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        logits, _ = model(idx)
        _loss(logits, y).backward()
        opt.step()

    opt.zero_grad(set_to_none=True)
    logits, _ = model(idx)
    _loss(logits, y).backward()
    dead = [n for n, p in model.named_parameters()
            if n not in DEAD_BY_REFERENCE_BRANCH and p.grad is not None
            and float(p.grad.norm()) == 0.0]
    assert dead == [], f"这些参数梯度恒为零：{dead}"


def test_optimizer_covers_every_parameter():
    """每个参数都要进 optimizer。"""
    model = NanoRWKV(TINY)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    covered = sum(len(g["params"]) for g in opt.param_groups)
    assert covered == len(list(model.parameters()))


def test_forward_is_causal():
    """改第 t 个 token 不能影响位置 < t 的 logits（wkv7 是单向的）。"""
    model = NanoRWKV(TINY).eval()
    idx, _ = _batch(batch=1, ctx=12, vocab=16, seed=1)
    with torch.no_grad():
        base, _ = model(idx)
        changed = idx.clone()
        changed[0, -1] = (changed[0, -1] + 1) % 16
        after, _ = model(changed)
    assert torch.allclose(base[:, :-1], after[:, :-1], atol=1e-6)
    assert not torch.allclose(base[:, -1], after[:, -1])


def test_state_step_by_step_matches_full_forward():
    """逐 token 推理（带 state）必须与整段前向一致——这是生成路径正确性的根。"""
    model = NanoRWKV(TINY).eval()
    idx, _ = _batch(batch=2, ctx=24, vocab=32, seed=2)
    with torch.no_grad():
        full, _ = model(idx)

        state = None
        steps = []
        for t in range(idx.shape[1]):
            logits, state = model(idx[:, t:t + 1], state)
            steps.append(logits)
        incremental = torch.cat(steps, dim=1)

    assert torch.allclose(full, incremental, atol=1e-5, rtol=1e-4), (
        (full - incremental).abs().max().item()
    )


def test_state_is_fp32_and_detached():
    """wkv 状态必须是 fp32（推理存储口径），且不回传梯度。"""
    model = NanoRWKV(TINY)
    idx, y = _batch()
    logits, state = model(idx)
    _loss(logits, y).backward()
    for w in state.wkv_state:
        assert w.dtype == torch.float32
        assert not w.requires_grad
    assert state.att_prev[0].dtype == logits.dtype


def test_state_zeros_shape():
    """零状态的形状必须与 cfg 一致。"""
    state = RWKVState.zeros(3, TINY, torch.device("cpu"), torch.float32)
    assert len(state.att_prev) == len(state.ffn_prev) == len(state.wkv_state) == TINY.n_layer
    assert state.att_prev[0].shape == (3, TINY.n_embd)
    assert state.wkv_state[0].shape == (3, TINY.n_head, TINY.head_size, TINY.head_size)


def test_ctx_len_guard():
    """超过 ctx_len 的输入必须直接报错，不能静默算错。"""
    model = NanoRWKV(TINY)
    too_long = torch.zeros(1, TINY.ctx_len + 1, dtype=torch.long)
    with pytest.raises(ValueError):
        model(too_long)


def test_single_batch_overfits():
    """单 batch 必须能过拟合到接近 0：证明模型/损失/优化器这条链路是通的。"""
    torch.manual_seed(0)
    model = NanoRWKV(NanoConfig(n_layer=2, n_embd=32, dim_ffn=64))
    model.train()
    idx, y = _batch(batch=2, ctx=32, vocab=8, seed=3)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = None
    for step in range(400):
        opt.zero_grad(set_to_none=True)
        logits, _ = model(idx)
        loss = _loss(logits, y)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert first is not None and first > math.log(8) * 0.5
    assert last < 0.05, f"400 步没拟合下来：{first:.3f} -> {last:.3f}"
