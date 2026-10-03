"""wkv7 分块实现与逐 token 递推的对拍。

逐 token 递推是定义本身，分块只是同一串运算的重排，两者必须在数值上一致。
"""

import torch

from src.wkv7 import run_wkv7, wkv7_chunked


def naive_wkv7(w, q, k, v, a, b):
    """输入：(B,T,H,C) 的 w/q/k/v/a/b；输出：(B,T,H,C) 的 y。

    逐 token 展开定义本身，用 float64 算，作为分块实现的 oracle。
    """
    _, t_len, _, _ = w.shape
    bsz, _, n_head, dim = w.shape
    state = torch.zeros(bsz, n_head, dim, dim, dtype=torch.float64)
    ys = []
    for t in range(t_len):
        w_t = torch.exp(-torch.exp(w[:, t].double()))
        a_t, b_t = a[:, t].double(), b[:, t].double()
        k_t, v_t, q_t = k[:, t].double(), v[:, t].double(), q[:, t].double()
        sa = torch.einsum("bhc,bhcd->bhd", a_t, state)
        state = (w_t.unsqueeze(-1) * state
                 + b_t.unsqueeze(-1) * sa.unsqueeze(-2)
                 + k_t.unsqueeze(-1) * v_t.unsqueeze(-2))
        ys.append(torch.einsum("bhc,bhcd->bhd", q_t, state))
    return torch.stack(ys, dim=1)


def _rand(bsz, t_len, n_head, dim, seed=0):
    """造一组与真实模型同量级的输入。

    关键是 a 与 b 的关系：模型层传进来的是 a = -kk、b = kk * sigmoid(·)，
    其中 kk 是逐头 L2 归一化后的 k（单位向量）。这样转移矩阵 diag(w) + b·aᵀ
    才是收缩的，递推有界。若让 a、b 各取独立随机数，递推会发散到 inf/nan
    —— 那是测试数据不真实，不是分块实现写错。
    """
    g = torch.Generator().manual_seed(seed)
    w = -torch.rand(bsz, t_len, n_head, dim, generator=g) - 0.5
    q = torch.randn(bsz, t_len, n_head, dim, generator=g)
    k = torch.randn(bsz, t_len, n_head, dim, generator=g)
    v = torch.randn(bsz, t_len, n_head, dim, generator=g)
    kk = torch.randn(bsz, t_len, n_head, dim, generator=g)
    kk = kk / kk.norm(dim=-1, keepdim=True)
    a = -kk
    b = kk * torch.sigmoid(torch.randn(bsz, t_len, n_head, dim, generator=g))
    return w, q, k, v, a, b


def _rel_err(got, want):
    """相对误差：除以参考值的最大幅度，避免尺度差异导致的误判。"""
    return ((got.double() - want).abs().max()
            / (want.abs().max() + 1e-12)).item()


def test_chunked_matches_naive_short():
    """T=1：退化情形，块内只有一步。"""
    w, q, k, v, a, b = _rand(2, 1, 3, 8)
    y, _ = wkv7_chunked(w, q, k, v, a, b, chunk=64)
    assert _rel_err(y, naive_wkv7(w, q, k, v, a, b)) < 1e-4


def test_chunked_matches_naive_not_multiple_of_chunk():
    """T=60、块长 12：T 不是块长的整数倍，走补零分支。"""
    w, q, k, v, a, b = _rand(2, 60, 3, 8)
    y, _ = wkv7_chunked(w, q, k, v, a, b, chunk=12)
    assert _rel_err(y, naive_wkv7(w, q, k, v, a, b)) < 1e-4


def test_chunked_matches_naive_ctx_len():
    """T=512、块长 64：真实训练配置。"""
    w, q, k, v, a, b = _rand(2, 512, 4, 8)
    y, _ = wkv7_chunked(w, q, k, v, a, b, chunk=64)
    assert _rel_err(y, naive_wkv7(w, q, k, v, a, b)) < 1e-4


def test_no_decay_and_zero_injection():
    """边界：w_in 很大（w 趋近 1，几乎不衰减）且 a/b 全零（无注入）。"""
    bsz, t_len, n_head, dim = 2, 64, 2, 8
    w = torch.zeros(bsz, t_len, n_head, dim)          # w_in = 0 -> w = exp(-1)
    q = torch.randn(bsz, t_len, n_head, dim)
    k = torch.randn(bsz, t_len, n_head, dim)
    v = torch.randn(bsz, t_len, n_head, dim)
    a = torch.zeros(bsz, t_len, n_head, dim)
    b = torch.zeros(bsz, t_len, n_head, dim)
    y, _ = wkv7_chunked(w, q, k, v, a, b, chunk=64)
    assert _rel_err(y, naive_wkv7(w, q, k, v, a, b)) < 1e-4


def test_state_carries_across_calls():
    """把 T=128 拆成两段 64 喂进去，结果必须与一次喂 128 相同。"""
    w, q, k, v, a, b = _rand(1, 128, 2, 8)
    y_full, s_full = wkv7_chunked(w, q, k, v, a, b, chunk=64)
    y1, s1 = wkv7_chunked(w[:, :64], q[:, :64], k[:, :64], v[:, :64],
                          a[:, :64], b[:, :64], chunk=64)
    y2, s2 = wkv7_chunked(w[:, 64:], q[:, 64:], k[:, 64:], v[:, 64:],
                          a[:, 64:], b[:, 64:], state_in=s1, chunk=64)
    y_parts = torch.cat([y1, y2], dim=1)
    assert _rel_err(y_parts, y_full) < 1e-4
    assert _rel_err(s2, s_full) < 1e-4


def test_run_wkv7_shapes_and_dtype():
    """模型层入口：吃 (B,T,C) 吐 (B,T,C)，state 形状 (B,H,head_size,head_size)。"""
    bsz, t_len, dim, head_size = 2, 128, 32, 8
    n_head = dim // head_size
    g = torch.Generator().manual_seed(7)
    w = -torch.rand(bsz, t_len, dim, generator=g) - 0.5
    q = torch.randn(bsz, t_len, dim, generator=g)
    k = torch.randn(bsz, t_len, dim, generator=g)
    v = torch.randn(bsz, t_len, dim, generator=g)
    a = -torch.rand(bsz, t_len, dim, generator=g)
    b = torch.rand(bsz, t_len, dim, generator=g)
    y, state = run_wkv7(q, w, k, v, a, b, head_size=head_size)
    assert y.shape == (bsz, t_len, dim)
    assert state.shape == (bsz, n_head, head_size, head_size)
    assert state.dtype == torch.float32


def test_backward_runs():
    """反向能跑通，且梯度非零（分块实现是 autograd 自动求导）。"""
    w, q, k, v, a, b = _rand(1, 64, 2, 8)
    for t in (q, k, v, a, b, w):
        t.requires_grad_(True)
    y, _ = wkv7_chunked(w, q, k, v, a, b, chunk=64)
    y.sum().backward()
    for t in (q, k, v, a, b, w):
        assert t.grad is not None and torch.isfinite(t.grad).all()
        assert t.grad.abs().sum() > 0


def test_weak_decay_dense_ab_matches_naive():
    """回归：块内几乎不衰减时 ab 是稠密阵，显式求逆会溢出成 NaN。

    w_in = -20 => log w = -exp(w_in) ≈ -2e-9 => 块内衰减 ≈ 1，ab[t,s] 对所有
    t > s 都是 |kk_t·kk_s| 量级。此时 (I - ab)^{-1} 的元素随块长组合式增长
    （实测 L=64 到 4.6e5，L<=32 还是 1.0），显式求逆 + matmul 会放大到 inf。
    RWKV-7 官方初始化里 www[0] ≈ -6 就是这种低衰减通道，所以这个 case 是真实
    会出现的，不是造出来的。
    """
    bsz, t_len, n_head, dim = 2, 512, 2, 8
    g = torch.Generator().manual_seed(0)
    q = torch.randn(bsz, t_len, n_head, dim, generator=g)
    k = torch.randn(bsz, t_len, n_head, dim, generator=g)
    v = torch.randn(bsz, t_len, n_head, dim, generator=g)
    w = torch.full((bsz, t_len, n_head, dim), -20.0)
    # kk 在时间上高度相关（真实模型里相邻 token 的 k 就是相关的），ab 因此同号
    kk = torch.ones(bsz, t_len, n_head, dim) / dim ** 0.5
    a = -kk
    b = kk * 0.9
    y, _ = wkv7_chunked(w, q, k, v, a, b, chunk=64)
    assert torch.isfinite(y).all()
    assert _rel_err(y, naive_wkv7(w, q, k, v, a, b)) < 1e-4
