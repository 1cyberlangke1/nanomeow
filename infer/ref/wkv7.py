"""定点 wkv7 递推：state 存 int32 网格，decay 走 Q15 网格。

逐 token 形式（每个 (head) 维护一个 N×N 的状态 S，行 = key 维、列 = value 维）：

    sa_t   = a_t^T S_{t-1}
    S_t    = diag(w_t) S_{t-1} + b_t sa_t + k_t v_t^T
    y_t    = q_t^T S_t          # 用的是**更新后**的 S_t，含 t 自己那次注入

训练侧（`train/src/wkv7.py`）用分块形式算同一件事，QAT 打开时插了四个点：

1. q / k / v / a / b 各走一次 per-tensor 动态 int8（`a` / `b` 是 float 乘积，要再量化一次）；
2. decay 走 Q15 网格：`w_lut = round(exp(log_w) * 2^15) * 2^-15`，`log_w = -exp(w_in)`；
3. `state_step = s_k * s_v`，就是 k·vᵀ 累加器的最小单位；
4. 每块末把 state round 到 `state_step` 的网格上。

本文件把 3 / 4 合并成「state 直接以 step 为单位存整数」：这样每次注入都是整数加法，
状态天然落在网格上（逐 token round 比训练侧逐块 round 更贴网格，只差半格量化）。

单位约定：state 的 1 个单位 = step = s_k * s_v。
    sa_j     = sum_i a_code[i] * S[i][j]                   （单位：s_a * step）
    inj_b    = apply_scale(b_code[i] * sa_j, s_b * s_a)    （落到 step 单位）
    inj_kv   = k_code[i] * v_code[j]                       （天然就是 step 单位）
    decayed  = round_div(S[i][j] * w15, 1 << 15)           （w 是 Q15）
    y_j      = sum_i q_code[i] * S[i][j]                   （scale = s_q * s_k * s_v）
"""

from .fixed import apply_scale, div, mul, round_div
from .int8_model import QTensor
from .nonlinear import ONE, exp_q, from_fixed, to_fixed
from .norm import isqrt


def normalize_p2(x, groups):
    """输入：QTensor x（长度 = groups * 组大小）、组数（展平的 (token, head) 对数）；输出：QTensor。

    预期行为：复刻 `fq_act(F.normalize(x.view(..., groups, -1), dim=-1, p=2))`：
              逐组算 L2 范数（平方和是 Q32，开方直接得到 Q16），用范数除每个元素，
              最后 per-tensor 动态量化回 int8。范数为 0 的组输出全 0 ——
              F.normalize 里 `max(norm, eps)` 在 eps = 1e-12 下就是这个效果。
    """
    n = len(x.codes)
    assert groups >= 1 and n % groups == 0, "长度必须能被组数整除"
    size = n // groups
    v = to_fixed(x)
    out = []
    for gi in range(groups):
        base = gi * size
        sos = 0
        for i in range(size):
            sos += v[base + i] * v[base + i]
        norm_q16 = isqrt(sos)
        if norm_q16 == 0:
            out.extend([0] * size)
            continue
        for i in range(size):
            out.append(round_div(v[base + i] * ONE, norm_q16))
    return from_fixed(out)


def decay_q15(w):
    """输入：w_in 的 QTensor（训练侧 `w = fq_act(-softplus(-decay_in) - 0.5)`，恒 <= -0.5）；
    输出：Q15 整数码列表，值 = round(exp(-exp(w_in)) * 2^15)，与训练侧 `w_lut` 同网格。

    预期行为：log_w = -exp(w_in) ∈ [-0.6065, 0)，w = exp(log_w) ∈ [0.545, 1)；
              两次 exp 都只吃非正数，正好落在 exp_q 的定义域里。
              w_real = w_q16 / 2^16，乘 2^15 就是除 2。
    """
    out = []
    for val in to_fixed(w):
        log_w = -exp_q(val)
        out.append(round_div(exp_q(log_w), 2))
    return out


def wkv7_recurrence(q, k, v, a, b, w15, state, head_size, step_prev=None):
    """输入：q / k / v / a / b 各是 QTensor（长度 T*C，C = n_head * head_size）；
              w15 是长度 T*C 的 Q15 整数码（`decay_q15` 的输出）；
              state 是 (n_head, head_size, head_size) 的整数列表（**就地更新**）；
              step_prev 是 state 里整数当前用的单位（上一个 token 的 s_k * s_v），
              None 表示 state 是全零（首 token）；
    输出：y 的 QTensor，长度 T*C，scale = s_q * s_k * s_v。

    预期行为：逐 token 逐 head 跑「sa → 更新 state → y」，全部整数运算；
              state 的单位是 step = s_k * s_v，落在训练侧 int32 网格上。
              训练侧每个 chunk 边界都做 `state = round(state / state_step) * state_step`，
              而逐 token 推理时每个 token 的 step 都不同（per-tensor 动态），所以旧 state
              必须先从 step_prev 换算到当前 step 再参与递推，否则会系统性偏大。
    """
    c = len(q.codes)
    n_head = len(state)
    n = head_size
    assert c == len(k.codes) == len(v.codes) == len(a.codes) == len(b.codes) == len(w15)
    assert c % (n_head * n) == 0, "通道数必须是 n_head * head_size 的整数倍"
    t_len = c // (n_head * n)
    step = mul(k.scale, v.scale)
    if step_prev is not None:
        # 换单位：state 实值不变，只是整数单位从 step_prev 变成 step，
        # 取整口径与训练侧 `_Round` 一致（四舍六入五成双）。
        ratio = div(step_prev, step)
        for h in range(n_head):
            s = state[h]
            for i in range(n):
                row = s[i]
                for j in range(n):
                    row[j] = apply_scale(row[j], ratio)
    ab_scale = mul(b.scale, a.scale)
    ys = [0] * c
    for t in range(t_len):
        base = t * n_head * n
        for h in range(n_head):
            head_base = base + h * n
            s = state[h]
            a_codes = a.codes[head_base:head_base + n]
            b_codes = b.codes[head_base:head_base + n]
            k_codes = k.codes[head_base:head_base + n]
            v_codes = v.codes[head_base:head_base + n]
            q_codes = q.codes[head_base:head_base + n]
            w_codes = w15[head_base:head_base + n]
            sa = [0] * n
            for i in range(n):
                ai = a_codes[i]
                if ai:
                    row = s[i]
                    for j in range(n):
                        sa[j] += ai * row[j]
            for i in range(n):
                row = s[i]
                w_i = w_codes[i]
                bi = b_codes[i]
                ki = k_codes[i]
                for j in range(n):
                    val = round_div(row[j] * w_i, 1 << 15)
                    if bi and sa[j]:
                        val += apply_scale(bi * sa[j], ab_scale)
                    if ki:
                        val += ki * v_codes[j]
                    row[j] = val
            for j in range(n):
                acc = 0
                for i in range(n):
                    acc += q_codes[i] * s[i][j]
                ys[head_base + j] = acc
    return QTensor(ys, mul(q.scale, step))
