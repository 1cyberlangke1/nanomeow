"""INT8 纯整数前向：逐 token、state 续接。

输入：`weights.load_weights` 给出的 `(NanoConfig, dict[名字] -> QMatrix | QTensor)`。
输出：`forward_token(token, states)` 返回 logits 的 QTensor（长度 V）。
预期行为：与 `train/src/model.py` 打开 QAT 后的前向同构 —— 每个 `fq_act` 点对应一次
          per-tensor 动态量化，每个 QATLinear 对应一次 `linear()`（权重 per-row）。
          把同一串 token 一个一个喂给训练侧模型（T=1）应当得到同一份 logits。
          全程只有整数运算，state 以 int32 存在 `LayerState.wkv` 里。
"""

from .fixed import mul
from .elemwise import (
    qt_add,
    qt_add_real,
    qt_addmul,
    qt_mul,
    qt_neg,
    qt_relu_sq,
    qt_sub,
    qt_sum_head,
)
from .int8_model import QTensor, linear, quantize


def fq(x):
    """输入：QTensor；输出：per-tensor 动态量化后的 QTensor。

    预期行为：就是训练侧的 `fq_act` —— 激活 per-tensor 动态 int8；
              对已经量化过的张量是幂等的（量化后的码一定顶到边界）。
    """
    return quantize(x.codes, x.scale)
from .nonlinear import qt_sigmoid, qt_softplus, qt_tanh
from .norm import norm_q
from .wkv7 import decay_q15, normalize_p2, wkv7_recurrence

LN_EPS = 1e-5        # nn.LayerNorm 的默认 eps（ln0 / ln1 / ln2 / ln_out）
LN_X_EPS = 64e-5     # 参考实现里 GroupNorm 的 eps（`!!! notice eps value !!!`）


class LayerState:
    """一层的跨 token 状态：time-shift 的上一 token 与 wkv7 的 int32 state。

    输入：head 数、每个 head 的边长、通道数。
    输出：可反复传回 `Int8Model.forward_token` 的状态对象。
    预期行为：`att_prev` / `ffn_prev` 是上一 token 的 (ln1 / ln2) 输出（QTensor），
              首个 token 前是 None（对应训练侧 ZeroPad2d 的「首行视作 0」）；
              `wkv` 是 [n_head][head_size][head_size] 的整数矩阵，`wkv_step` 是它当前
              的整数单位（= 上一个 token 的 s_k * s_v；None 表示还没写过、state 全零）。
    """

    __slots__ = ("att_prev", "ffn_prev", "wkv", "wkv_step")

    def __init__(self, n_head, head_size):
        self.att_prev = None
        self.ffn_prev = None
        self.wkv = [[[0] * head_size for _ in range(head_size)] for _ in range(n_head)]
        self.wkv_step = None


class Int8Model:
    """逐 token 的 INT8 纯整数模型。

    输入：`cfg`（NanoConfig）与 `weights`（`weights.load_weights` 的输出）。
    输出：`forward_token` 返回 logits 的 QTensor；`zeros_state` 造初始状态。
    """

    def __init__(self, cfg, weights):
        self.cfg = cfg
        self.w = weights
        self.n_layer = cfg.n_layer
        self.n_head = cfg.n_head
        self.head_size = cfg.head_size

    def zeros_state(self):
        """输入：无；输出：`n_layer` 个全新的 LayerState（wkv 全零、time-shift 为空）。"""
        return [LayerState(self.n_head, self.head_size) for _ in range(self.n_layer)]

    def _norm(self, x, name, eps, groups):
        """输入：QTensor、参数名前缀、eps、组数；输出：QTensor。

        预期行为：对应训练侧 `fq_act(F.layer_norm(...))` / `fq_act(F.group_norm(...))`，
                  weight / bias 走 per-tensor 的 QTensor。
        """
        return norm_q(x, self.w[name + ".weight"], self.w[name + ".bias"], eps, groups)

    def _lowrank2(self, x, prefix):
        """输入：QTensor、低秩对的名字前缀（如 `blocks.0.att.v`）；输出：QTensor。

        预期行为：对应训练侧的 `lowrank2`，即两次 `matmul_w`。
        """
        return linear(linear(x, self.w[prefix + "1"]), self.w[prefix + "2"])

    def _shift(self, prev, x):
        """输入：上一 token 的 QTensor（或 None）、当前 token 的 QTensor；输出：QTensor。

        预期行为：对应训练侧 `fq_act(token_shift(...))`，即 `prev - x`；prev 为 None 时
                  首行按 0 处理，所以 delta = -x。
        """
        if prev is None:
            return fq(qt_neg(x))
        return fq(qt_sub(prev, x))

    def _embed(self, token):
        """输入：token id；输出：该行权重（per-row 量化）的 QTensor。

        预期行为：对应 `FakeQuantizedEmbedding` —— 权重 per-row 量化后查表，
                  查表本身不做激活量化。
        """
        mat = self.w["emb.weight"]
        base = token * mat.cols
        return QTensor(mat.codes[base:base + mat.cols], mat.scales[token])

    def _att(self, i, h1, att_prev, v_first, st):
        """输入：层号、ln1 输出、上一 token 的 att 输入、v_first、该层 LayerState（就地更新）；
        输出：(att 输出, 新的 v_first)。

        预期行为：逐行复刻训练侧 `RWKV_Tmix_x070.forward` 在 T=1 下的计算，
                  顺序与插桩点一一对应。
        """
        p = f"blocks.{i}.att."
        xx = self._shift(att_prev, h1)
        xr = fq(qt_addmul(h1, xx, self.w[p + "x_r"]))
        xw = fq(qt_addmul(h1, xx, self.w[p + "x_w"]))
        xk = fq(qt_addmul(h1, xx, self.w[p + "x_k"]))
        xv = fq(qt_addmul(h1, xx, self.w[p + "x_v"]))
        xa = fq(qt_addmul(h1, xx, self.w[p + "x_a"]))
        xg = fq(qt_addmul(h1, xx, self.w[p + "x_g"]))

        # 四路 QATLinear（receptance / key / value / output）只量化输入与权重，**不量化输出**：
        # 量化发生在后面显式的 fq 点或 wkv 递推入口，所以这里一律 quantize=False 拿原始值。
        r = linear(xr, self.w[p + "receptance.weight"], quantize=False)
        decay_in = qt_add(
            self.w[p + "w0"],
            linear(qt_tanh(linear(xw, self.w[p + "w1"])), self.w[p + "w2"]),
            quantize=False)
        w = fq(qt_add_real(qt_neg(qt_softplus(qt_neg(decay_in), quantize=False)), -1, 2))

        k = linear(xk, self.w[p + "key.weight"], quantize=False)
        v = linear(xv, self.w[p + "value.weight"], quantize=False)
        if i == 0:
            v_first = v
        else:
            gate = fq(qt_sigmoid(
                qt_add(self.w[p + "v0"], self._lowrank2(xv, p + "v"), quantize=False)))
            v = fq(qt_addmul(v, qt_sub(v_first, v, quantize=False), gate))

        a = fq(qt_sigmoid(
            qt_add(self.w[p + "a0"], self._lowrank2(xa, p + "a"), quantize=False)))
        g = linear(fq(qt_sigmoid(linear(xg, self.w[p + "g1"]))), self.w[p + "g2"])

        # kk 用**原始** k（训练侧 `fq_act(k * k_k)`）；k 的更新用**已量化**的 k
        # （训练侧 `fused_k_rwkv7(fq_act(k), ...)`），两者不能混。
        kk = normalize_p2(fq(qt_mul(k, self.w[p + "k_k"])), self.n_head)
        k_q = fq(k)
        k = fq(qt_addmul(
            k_q, k_q, qt_mul(qt_add_real(a, -1, quantize=False), self.w[p + "k_a"],
                          quantize=False)))

        # 递推就地更新 st.wkv；state 的整数单位随 token 变，靠 st.wkv_step 记住上一趟的。
        # q/k/v/a/b 是递推本体的插桩点：q、v 在这里才第一次量化，k/a/b 已经是量化值。
        # a = -kk 必须**再量化一次**：训练侧 run_wkv7 里是 `a = qat(a)`，而
        # `max(mx*128, |mn|*127)` 对取负不对称（127 / 128 两端不同口径），
        # 直接取负会沿用 kk 的 scale、和训练侧差 128/127 ≈ 0.79%，state 会跑飞。
        v_q = fq(v)
        y = wkv7_recurrence(
            fq(r), k, v_q, fq(qt_neg(kk)), qt_mul(kk, a, quantize=True), decay_q15(w),
            st.wkv, self.head_size, st.wkv_step)
        st.wkv_step = mul(k.scale, v_q.scale)
        y = self._norm(y, p + "ln_x", LN_X_EPS, self.n_head)
        # 残差项用的是**原始** r 与递推后的 k（训练侧没有对 r 再量化）
        rk = fq(qt_mul(qt_mul(r, k, quantize=False), self.w[p + "r_k"], quantize=True))
        y = fq(qt_add(y, qt_mul(qt_sum_head(rk, self.head_size), v, quantize=False)))
        return linear(fq(qt_mul(y, g)), self.w[p + "output.weight"], quantize=False), v_first

    def _ffn(self, i, h2, ffn_prev):
        """输入：层号、ln2 输出、上一 token 的 ffn 输入；输出：CMix 输出。"""
        p = f"blocks.{i}.ffn."
        xx = self._shift(ffn_prev, h2)
        k = fq(qt_addmul(h2, xx, self.w[p + "x_k"]))
        # CMix 的 key / value 同样是 QATLinear：量化点是 `fq_act(relu(key(k)) ** 2)`
        # 与 Block 里残差相加后的 fq_act，所以这里都取原始值。
        k = fq(qt_relu_sq(linear(k, self.w[p + "key.weight"], quantize=False)))
        return linear(k, self.w[p + "value.weight"], quantize=False)

    def forward_token(self, token, states):
        """输入：token id、`zeros_state()` 造的状态列表（**就地更新**）；输出：logits 的 QTensor。

        预期行为：一层一层跑 ln0 -> att 残差 -> ffn 残差，最后 ln_out + head；
                  与训练侧「一个 token 一次前向」的结果同构。
        """
        x = self._embed(token)
        v_first = None
        for i in range(self.n_layer):
            st = states[i]
            if i == 0:
                x = self._norm(x, "blocks.0.ln0", LN_EPS, 1)
            h1 = self._norm(x, f"blocks.{i}.ln1", LN_EPS, 1)
            att_out, v_first = self._att(i, h1, st.att_prev, v_first, st)
            st.att_prev = h1
            x = fq(qt_add(x, att_out))

            h2 = self._norm(x, f"blocks.{i}.ln2", LN_EPS, 1)
            ffn_out = self._ffn(i, h2, st.ffn_prev)
            st.ffn_prev = h2
            x = fq(qt_add(x, ffn_out))

        x = self._norm(x, "ln_out", LN_EPS, 1)
        return linear(x, self.w["head.weight"])
