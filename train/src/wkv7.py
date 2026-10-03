"""RWKV-7 的 WKV 递推：纯 PyTorch 分块实现（不调任何现成算子）。

逐 token 形式，每个 (batch, head) 维护一个 C×C 状态 S（行 j = key 维，列 i = value 维）：

    sa_t = a_t^T S_{t-1}
    S_t  = diag(w_t) S_{t-1} + b_t sa_t + k_t v_t^T
    y_t  = q_t^T S_t

其中 w_t = exp(-exp(w_in_t))。合并前两项得
S_t = (diag(w_t) + b_t a_t^T) S_{t-1} + k_t v_t^T，转移矩阵是「对角 + 秩一」，
这就是 RWKV-7 的广义 delta rule。

分块形式（块长 L）：记块内累积衰减 LW_t = Σ_{s<=t} log w_s（log w = -exp(w_in)，
不需要再取对数），并令

    q̃_t = q_t·e^{LW_t}   ã_t = a_t·e^{LW_{t-1}}   k̂_s = k_s·e^{-LW_s}   b̂_s = b_s·e^{-LW_s}

则块内所有 sa 满足一个严格下三角线性方程组

    SA = Ã S_0 + tril(Ã B̂ᵀ, -1) SA + tril(Ã K̂ᵀ, -1) V

解出 SA 之后，y 和块末状态都只是矩阵乘：

    Y   = Q̃ S_0 + tril(Q̃ B̂ᵀ, 0) SA + tril(Q̃ K̂ᵀ, 0) V
    S_L = diag(e^{LW_L}) (S_0 + B̂ᵀ SA + K̂ᵀ V)

块间只剩 T/L 次串行迭代。

块内 SA 由 (I - ab) SA = rhs 解出，**解算方式决定数值稳定性**。ab 是块内 token
的两两相互作用（ab[t,s] = a_t·b_s·e^{LW_{t-1} - LW_s}，t > s）。衰减弱时
（w_in 很负 => log w = -e^{w_in} ≈ 0 => 块内几乎不衰减）ab 会退化成稠密的严格
下三角阵，而 (I - ab)^{-1} = Σ_{i<L} ab^i 的元素随块长组合式增长。用 step5000
权重扫 40 批真实窗口实测（max|ab| 全程 ≈ 0.70）：

    L = 8/16/24/32 -> max|inv| = 1.0
    L = 48         -> max|inv| = 1.8e2
    L = 64         -> max|inv| = 4.6e5

早期这里显式求逆再 matmul（tri_inv @ rhs），中间量先被放大到 1e5、再靠相消回到
真实值：fp32 下直接溢出，L=64 时 40/40 批前向出 NaN，训练里 loss 从 1752 步起
永久非有限；反向也被 tri_inv 的乘积放大到 |grad| ~ 1e5，梯度裁剪把学习信号压成
噪声，loss 于是卡在 unigram 水平（实测 ~4.3）。现在改成三角求解
（torch.linalg.solve_triangular）：数学上完全等价，但后向稳定，中间量始终是解
本身的量级 —— 同一 checkpoint 下 L = 16..128 全部 0/40 非有限，logits 与块长
无关（回归测试见 tests/test_wkv7.py::test_weak_decay_dense_ab_matches_naive）。

b̂/k̂ 里的 e^{-LW} 不是瓶颈：反向精度与 float64 逐 token oracle 对拍，L = 8..128
的相对误差都在 1e-7。

块长上界因此只剩**精度**意义：三角求解的误差 ~ cond(I-ab)·eps，而 cond 随块长
组合式增长。参考实现的 CUDA kernel 用 CHUNK_LEN = 16，这里对齐它：CHUNK = 16，
NanoConfig.wkv_chunk 也取 16。

全程真 fp32（`autocast(enabled=False)`）：块内解 (I - ab) SA = rhs 是三角求解，
bf16 只有 8 位尾数，解病态三角系统时误差会被放大。模型其余部分照常走 bf16，
这里是唯一的例外。
"""

import torch

from .qat import _Round

CHUNK = 16

# decay 的 exp 走 256 项 × 2 字节的定点 LUT，输出是 16 位定点（Q15），
# 所以 QAT 里按 2^-15 的网格量化 w = exp(log_w)。
WKV_LUT_FRAC_BITS = 15


def to_chunks(t: torch.Tensor, n_chunk: int, chunk: int) -> torch.Tensor:
    """输入：(B,T,H,C)、块数、块长；输出：(B,H,n_chunk,chunk,C)。

    把时间维切成 (n_chunk, chunk) 并把 head 提到前面，让块内矩阵乘的
    batch 维是 (B,H,n_chunk)。
    """
    b, _, h, c = t.shape
    return t.view(b, n_chunk, chunk, h, c).permute(0, 3, 1, 2, 4)


def wkv7_chunked(w, q, k, v, a, b, state_in=None, chunk=CHUNK, qat=None):
    """WKV-7 前向（纯 PyTorch，反向由 autograd 生成）。

    输入：
        w,q,k,v,a,b: (B,T,H,C)。w 是**未经变换的** w_in（内部做 exp(-exp(·))）；
                     a 是缩并向量、b 是外积向量（模型层传 a=-kk、b=kk*a_model）。
        state_in:    (B,H,C,C) fp32 或 None（None 视作零状态）。
        chunk:       块长，不整除 T 时自动补零。
        qat:         IntxFakeQuantizer 或 None。给定时把递推本体的定点边界建模进去
                     （q/k/v/a/b 走 int8、decay 走 Q15 定点 LUT、state 走 int32 网格）；
                     None 或 disabled 时与不开量化逐位一致。
    输出：
        y:          (B,T,H,C)，dtype 跟随 v。
        state_out:  (B,H,C,C) fp32。
    """
    with torch.autocast(device_type=w.device.type, enabled=False):
        return _wkv7_chunked_fp32(w, q, k, v, a, b, state_in, chunk, qat)


def _wkv7_chunked_fp32(w, q, k, v, a, b, state_in, chunk, qat=None):
    """wkv7_chunked 的计算主体；调用方保证已经关掉 autocast。参数同上。"""
    bsz, t_len, n_head, dim = w.shape
    out_dtype = v.dtype

    # 补齐到块长整数倍。补的位置必须是「什么都不发生」：q/k/v/a/b 补 0 表示既无注入
    # 也无缩并；log 衰减补 0 表示 w = 1、状态原样通过。注意补在 log 域而不是 w_in 域
    # —— w_in 要取到 w = 1 得是 -inf，没法用数值表示。
    pad = (-t_len) % chunk
    log_w = -torch.exp(w.float())
    if pad:
        log_w = torch.nn.functional.pad(log_w, (0, 0, 0, 0, 0, pad))
        q, k, v, a, b = (torch.nn.functional.pad(t.float(), (0, 0, 0, 0, 0, pad))
                         for t in (q, k, v, a, b))
    else:
        q, k, v, a, b = (t.float() for t in (q, k, v, a, b))
    t_pad = t_len + pad

    qat_on = qat is not None and qat.enabled
    if qat_on:
        # 递推本体的输入侧插桩。q/k/v/a/b 是激活，先量化到 int8；decay 的
        # exp 走定点 LUT，按 Q15 输出网格量化。w = exp(log_w) 恒在 [0.545, 1) 内
        # （w_in <= -0.5），所以这里的 clamp 永远不会触底，只是防 log(0)。
        q = qat(q)
        k = qat(k)
        k_scale = qat.scale
        v = qat(v)
        v_scale = qat.scale
        a, b = qat(a), qat(b)
        w_lut = torch.exp(log_w)
        w_lut = _Round.apply(w_lut * (2.0 ** WKV_LUT_FRAC_BITS)) * (2.0 ** -WKV_LUT_FRAC_BITS)
        log_w = torch.log(torch.clamp(w_lut, min=2.0 ** -WKV_LUT_FRAC_BITS))
        # state 存 int32：量化步长就是 k·vᵀ 累加器的最小单位 scale_k × scale_v，
        # 直接取量化器这一趟用的 scale，不在本地按范围重算。
        state_step = (k_scale * v_scale).clamp_min(1e-12)
    else:
        state_step = None

    # CUDA 快路径。分块实现把 [B,H,n,chunk,chunk] 的中间量物化到显存，一步读写几十 GB，
    # 显存带宽成瓶颈（实测 batch 768 反而比 256 慢）；参考 kernel 每个线程持 state 的一列、
    # token 向量走 shared，一步只读写一遍。kernel 只编了 head_size=8 / chunk=16，
    # 其余维度与 CPU 走下面的分块实现。
    if state_in is None and log_w.is_cuda and dim == 8 and chunk == 16:
        from .wkv7_cuda import wkv7_cuda
        y, state_out = wkv7_cuda(log_w, q.to(torch.bfloat16), k.to(torch.bfloat16),
                                 v.to(torch.bfloat16), a.to(torch.bfloat16),
                                 b.to(torch.bfloat16), chunk,
                                 state_step if state_step is not None else 0.0,
                                 return_state=True)
        if pad:
            y = y[:, :t_len]
        return y.to(out_dtype), state_out

    n_chunk = t_pad // chunk
    log_w, q, k, v, a, b = (to_chunks(t, n_chunk, chunk) for t in (log_w, q, k, v, a, b))

    # 块内累积衰减。cum 含当前位置，cum_prev 是进入该位置之前的累积
    # （ã 用 LW_{t-1}：sa_t 缩并的是**更新前**的状态 S_{t-1}）。
    tri_cum = torch.ones(chunk, chunk, device=log_w.device, dtype=log_w.dtype).tril()
    cum = tri_cum @ log_w
    cum_prev = cum - log_w
    q_t = q * torch.exp(cum)
    a_t = a * torch.exp(cum_prev)
    k_h = k * torch.exp(-cum)
    b_h = b * torch.exp(-cum)
    decay_chunk = torch.exp(cum[..., -1, :])          # (B,H,n,C)

    # 块内 token 对之间的相互作用。tril(-1) 是严格下三角（sa_t 只看 s<t 的注入），
    # tril(0) 含对角（y_t 用的是**更新后**的 S_t，包含 t 自己那次注入）。
    ab = a_t @ b_h.transpose(-1, -2)
    ak = a_t @ k_h.transpose(-1, -2)
    qb = q_t @ b_h.transpose(-1, -2)
    qk = q_t @ k_h.transpose(-1, -2)
    tri_lo = torch.ones(chunk, chunk, device=w.device, dtype=torch.bool).tril(-1)
    tri_diag = torch.ones(chunk, chunk, device=w.device, dtype=torch.bool).tril(0)
    ab = ab * tri_lo
    ak = ak * tri_lo
    qb = qb * tri_diag
    qk = qk * tri_diag
    # (I - ab) 是单位下三角（对角恒 1）。解 SA 用三角求解，**不显式求逆**：
    # 显式求逆会把中间量放大到 (I-ab)^{-1} 的量级、再靠相消回到真实值，fp32 下
    # 直接溢出成 NaN；三角求解后向稳定，中间量始终是解本身的量级。
    # 推导与实测数字见模块 docstring。
    unit_lower = torch.eye(chunk, device=ab.device, dtype=ab.dtype) - ab

    if state_in is None:
        state = torch.zeros(bsz, n_head, dim, dim, device=w.device, dtype=torch.float32)
    else:
        state = state_in.float()

    # 块间串行：每块解出 SA、算出 y、推进状态。块内已全部并行，这里只剩 T/L 次迭代。
    ys = []
    for c in range(n_chunk):
        from_state_a = a_t[:, :, c] @ state
        from_state_q = q_t[:, :, c] @ state
        rhs = from_state_a + ak[:, :, c] @ v[:, :, c]
        sa = torch.linalg.solve_triangular(unit_lower[:, :, c], rhs,
                                           upper=False, unitriangular=True)
        ys.append(from_state_q + qb[:, :, c] @ sa + qk[:, :, c] @ v[:, :, c])
        injected = (b_h[:, :, c].transpose(-1, -2) @ sa
                    + k_h[:, :, c].transpose(-1, -2) @ v[:, :, c])
        state = decay_chunk[:, :, c].unsqueeze(-1) * (state + injected)
        if qat_on:
            state = _Round.apply(state / state_step) * state_step

    y = torch.stack(ys, dim=2)                        # (B,H,n,L,C)
    y = y.permute(0, 2, 3, 1, 4).reshape(bsz, t_pad, n_head, dim)
    if pad:
        y = y[:, :t_len]
    # 必须 contiguous：permute 后 reshape 在 n_head == 1 时不会复制、直接返回非连续
    # 视图（H > 1 时才需要复制），pad 切片同样产生非连续视图。调用方按 (B,T,C) 连续
    # 布局用 .view 就会报 view size is not compatible。
    return y.contiguous().to(out_dtype), state


def run_wkv7(q, w, k, v, a, b, state_in=None, head_size=8, chunk=CHUNK, qat=None):
    """模型层入口：吃 (B,T,C)，吐 (B,T,C) 与 state。

    输入：(B,T,C) 的 q/w/k/v/a/b、可选 state_in (B,H,head_size,head_size)、
          可选 qat（递推本体插桩点，见 wkv7_chunked）。
    输出：y (B,T,C)、state_out (B,H,head_size,head_size)。

    state_out 切断梯度：它只在推理时逐 token 传递，训练时每个窗口都从零状态开始，
    不把梯度经由状态传回上一个窗口（BPTT 的截断点就在这里）。
    """
    out_dtype = q.dtype
    bsz, t_len, hidden = q.shape
    n_head = hidden // head_size
    q, w, k, v, a, b = (t.view(bsz, t_len, n_head, head_size) for t in (q, w, k, v, a, b))
    y, state_out = wkv7_chunked(w, q, k, v, a, b, state_in, chunk, qat)
    return y.reshape(bsz, t_len, hidden).to(out_dtype), state_out.detach()
