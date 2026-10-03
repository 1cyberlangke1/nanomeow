"""nanomeow 的 RWKV-7 x070 模型：`tmp/Mini_RWKV_7/src/model.py` 的逐段移植 + QAT 插桩。

移植原则：

- 算式、组件、初始化、常量逐段照抄参考，**一个组件都不砍，只缩维度**（参考里维度本来
  就由 args 驱动，这里换成 `NanoConfig`）。
- 只去掉参考的训练框架依赖（pytorch-lightning / deepspeed / rwkvfla / wandb）：
  `token_shift` / `fused_addcmul_rwkv7` / `fused_k_rwkv7` 换成 rwkvfla 的纯 torch
  等价写法，`RUN_CUDA_RWKV7g` 的纯 torch 分块版在 `src/wkv7.py`，另叠一条
  `src/wkv7_cuda.py` 的 CUDA 快路径（head_size=8 / chunk=16 且从零状态开始时启用，数值与分块版一致）。
- 新增参考没有的两样：`RWKVState`（逐 token 推理的跨步状态）与 QAT 插桩点。

QAT 怎么和「逐段移植」共存：

- 移植保持参考的算式一字不改；QAT 只是在**每一条定点边界**上套一层假量化
  （`IntxFakeQuantizer`），跟 torchao 把 `nn.Linear` 换成 `FakeQuantizedLinear` 是同一手法。
- 所有插桩点默认 `enabled=False`，此时是恒等映射，前向与参考**逐位一致**；
  `src/qat.py::prepare_qat()` 打开后，同一个计算图上的边界才变成 int8。

参考里假设 `dim_att == n_embd`（`x.view(B, T, n_head, -1)` 要能整除），本文件沿用。
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import NanoConfig
from .qat import IntxFakeQuantizer, per_row_int8, per_tensor_int8
from .wkv7 import run_wkv7

# 参考的 value residual 是**按层分支**——第 0 层只记录 `v_first`、不做插值
# （`if self.layer_id == 0: v_first = v`），所以第 0 层的 v0/v1/v2 结构性地拿不到梯度。
# 这里是**显式登记**（不是默默跳过）：白名单只允许这一个集合，
# 多出任何一个死参数都必须让测试挂掉。
DEAD_BY_REFERENCE_BRANCH = frozenset(
    f"blocks.0.att.{name}" for name in ("v0", "v1", "v2")
)


def _fq() -> IntxFakeQuantizer:
    """输入：无；输出：关闭状态的 per-tensor int8 假量化器（激活 / 逐元素常量插桩点）。

    预期行为：构造出来是 disabled 的，所以不开 QAT 时前向与参考逐位一致；
              `prepare_qat()` 会把全模型（含这里）的量化器一起打开。
    """
    return IntxFakeQuantizer(per_tensor_int8(), enabled=False)


def _fq_w() -> IntxFakeQuantizer:
    """输入：无；输出：关闭状态的 per-row int8 假量化器（裸权重矩阵插桩点）。

    预期行为：per-row 就是 per-channel（权重按输出通道对称 int8）。
    """
    return IntxFakeQuantizer(per_row_int8(), enabled=False)


def token_shift(pad, x, prev=None):
    """rwkvfla.modules.token_shift 的纯 torch 等价：返回 delta = x_prev - x。

    输入：pad 是 `nn.ZeroPad2d((0, 0, 1, -1))`、x (B,T,C)、prev (B,C) 或 None。
    输出：(B,T,C) 的 delta。
    预期行为：prev=None 时用 ZeroPad2d 造「首行视作 0」的移位（与参考逐位一致）；
              prev 给定时首行用它，供逐 token 推理跨步续接。
    """
    if prev is None:
        return pad(x) - x
    return torch.cat([prev.unsqueeze(1), x[:, :-1]], dim=1) - x


def fused_addcmul_rwkv7(x, xx, *params):
    """rwkvfla.ops.rwkv7.fused_addcmul 的纯 torch 等价。

    输入：x (B,T,C)、xx (B,T,C) 的 time-shift delta、6 个可广播系数。
    输出：列表，元素个数与系数个数相同。
    预期行为：对每个系数返回 x + xx * 系数（RWKV-7 官方式子）。
    """
    return [x + xx * p for p in params]


def fused_k_rwkv7(k, a, k_a):
    """rwkvfla.ops.rwkv7.fused_k_update 的纯 torch 等价。

    输入：k (B,T,C)、in-context 学习率 a (B,T,C)、k_a (1,1,C)。
    输出：更新后的 k，形状同 k。
    预期行为：k + k * (a - 1) * k_a。
    """
    return k + k * (a - 1) * k_a


class RWKV_Tmix_x070(nn.Module):
    """时间混合 + wkv7 递推（移植参考的 `RWKV_Tmix_x070`，只缩维度）。

    输入：forward(x, v_first, att_prev=None, wkv_state=None)；x (B,T,C) 是已过 ln1 的
          隐藏状态，v_first (B,T,C) 是 value residual 的参照向量（第 0 层由本层写入、
          layer_id>0 的层读它），att_prev (B,C) 是上一 chunk 的 time-shift 输入，
          wkv_state (B,H,N,N) fp32 是 wkv 状态。
    输出：(y (B,T,C), v_first_out (B,T,C), wkv_state_out (B,H,N,N) fp32)。
    """

    def __init__(self, cfg: NanoConfig, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id

        self.head_size = cfg.head_size
        self.wkv_chunk = cfg.wkv_chunk
        self.dim_lora = cfg.dim_lora
        self.n_head = cfg.dim_att // self.head_size
        assert cfg.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = cfg.n_embd

        ratio_0_to_1 = layer_id / (cfg.n_layer - 1)  # 0 to 1
        ratio_1_to_almost0 = 1.0 - (layer_id / cfg.n_layer)  # 1 to ~0
        ddd = torch.ones(1, 1, C)
        for i in range(C):
            ddd[0, 0, i] = i / C

        self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
        self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

        def ortho_init(x, scale):
            """参考原样移植的正交初始化：低秩对的上投影用它，下投影恒为 0。

            输入：待初始化张量 x（2D 或 3D）、缩放 scale。
            输出：原地初始化后的 x。
            """
            shape = x.shape
            if len(shape) == 2:
                gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                nn.init.orthogonal_(x, gain=gain * scale)
            elif len(shape) == 3:
                gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                for i in range(shape[0]):
                    nn.init.orthogonal_(x[i], gain=gain * scale)
            else:
                assert False
            return x

        www = torch.zeros(C)
        zigzag = torch.zeros(C)
        linear = torch.zeros(C)
        for n in range(C):
            linear[n] = n / (C - 1) - 0.5
            zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
            zigzag[n] = zigzag[n] * abs(zigzag[n])
            www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

        # 四组低秩对的 rank 都是 cfg.dim_lora。参考写的是
        #   D_DECAY_LORA = max(32, int(round((2.5 * (C ** 0.5)) * factor / 32) * 32))
        # （a/v/g 组系数分别是 2.5 / 1.7 / 5）。那个 max(32, ...) 是给大模型的地板：
        # 参考实际配置 C=512 / head_size=64 时四组都落在 32~64，即 D/C 约 1/8。
        # 缩维到 C=32 后照抄地板会得到 D=32=C —— 低秩对退化成满秩 32x32，
        # 相对容量是参考的 8 倍（实测能把每层 decay 压到数学下界 0.5452，
        # 等效记忆 ~2 token，模型因此完全不条件于 prompt），所以按同一比例取 D=8。
        D_DECAY_LORA = D_AAA_LORA = D_MV_LORA = D_GATE_LORA = self.dim_lora
        self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
        # !!! 0.5 comes from F.softplus !!!
        self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)

        self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
        self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
        self.a0 = nn.Parameter(torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4)

        self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
        self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
        self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)

        # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
        self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
        self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

        self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
        self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
        self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        # !!! notice eps value !!!
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

        self.receptance.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
        self.key.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
        self.value.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
        self.output.weight.data.zero_()
        del www, zigzag, linear, ddd

        # 插桩点（默认恒等，prepare_qat 才打开）：
        #   fq_act   —— token-shift 缓冲 / 6 路混合 / 非线性输出 / ln_x 输出 / 残差相加
        #   fq_param —— 逐元素常量 x_r…x_g、w0/a0/v0、k_k/k_a/r_k
        #   fq_w     —— 裸权重矩阵 w1/w2、a1/a2、v1/v2、g1/g2（per-row）
        #   fq_wkv   —— wkv7 递推本体（q/k/v/a/b 与 int32 state）
        self.fq_act = _fq()
        self.fq_param = _fq()
        self.fq_w = _fq_w()
        self.fq_wkv = _fq()

    def matmul_w(self, x, p):
        """输入：激活 x (..., K)、裸权重 p (K, N)；输出：假量化后的 x @ p。

        预期行为：p 按输出通道 per-row int8（p 的存储口径是 (K, N)，先转成 (N, K) 再走
                  fq_w）、输入与输出激活 per-tensor int8。
        """
        return self.fq_act(F.linear(self.fq_act(x), self.fq_w(p.transpose(0, 1))))

    def lowrank2(self, x, p1, p2):
        """输入：激活 x、裸权重对 p1 (C,D) / p2 (D,C)；输出：假量化后的 (x @ p1) @ p2。"""
        return self.matmul_w(self.matmul_w(x, p1), p2)

    def forward(self, x, v_first, att_prev=None, wkv_state=None):
        B, T, C = x.size()
        xx = self.fq_act(token_shift(self.time_shift, x, att_prev))
        xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(
            x, xx,
            self.fq_param(self.x_r), self.fq_param(self.x_w), self.fq_param(self.x_k),
            self.fq_param(self.x_v), self.fq_param(self.x_a), self.fq_param(self.x_g),
        )
        xr, xw, xk, xv, xa, xg = [self.fq_act(t) for t in (xr, xw, xk, xv, xa, xg)]

        r = self.receptance(xr)
        # soft-clamp to (-inf, -0.5)
        w = self.fq_act(
            -F.softplus(
                -(self.fq_param(self.w0)
                  + self.matmul_w(self.fq_act(torch.tanh(self.matmul_w(xw, self.w1))), self.w2))
            ) - 0.5
        )
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v  # store the v of the first layer
        else:
            v = self.fq_act(torch.lerp(
                v, v_first.to(v.dtype),
                self.fq_act(torch.sigmoid(
                    self.fq_param(self.v0) + self.lowrank2(xv, self.v1, self.v2))).to(v.dtype),
            ))  # add value residual
        # a is "in-context learning rate"
        a = self.fq_act(torch.sigmoid(
            self.fq_param(self.a0) + self.lowrank2(xa, self.a1, self.a2)))
        g = self.matmul_w(self.fq_act(torch.sigmoid(self.matmul_w(xg, self.g1))), self.g2)

        kk = self.fq_act(F.normalize(
            self.fq_act(k * self.fq_param(self.k_k)).view(B, T, self.n_head, -1),
            dim=-1, p=2.0)).view(B, T, C)
        k = self.fq_act(fused_k_rwkv7(self.fq_act(k), self.fq_act(a), self.fq_param(self.k_a)))

        x, wkv_state_out = run_wkv7(r, w, k, v, -kk, kk * a, wkv_state,
                                    self.head_size, chunk=self.wkv_chunk, qat=self.fq_wkv)
        x = self.fq_act(F.group_norm(
            x.view(B * T, C), self.n_head,
            self.fq_param(self.ln_x.weight), self.fq_param(self.ln_x.bias),
            self.ln_x.eps)).view(B, T, C)

        x = self.fq_act(x + (
            self.fq_act(r.view(B, T, self.n_head, -1) * k.view(B, T, self.n_head, -1)
                        * self.fq_param(self.r_k)).sum(dim=-1, keepdim=True)
            * v.view(B, T, self.n_head, -1)
        ).view(B, T, C))
        x = self.output(self.fq_act(x * g))
        return x, v_first, wkv_state_out


class RWKV_CMix_x070(nn.Module):
    """通道混合（移植参考的 `RWKV_CMix_x070`，只缩维度）。

    输入：forward(x, ffn_prev=None)；x (B,T,C) 是已过 ln2 的隐藏状态，
          ffn_prev (B,C) 是上一 chunk 的 time-shift 输入。
    输出：(B,T,C)。
    """

    def __init__(self, cfg: NanoConfig, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        ratio_1_to_almost0 = 1.0 - (layer_id / cfg.n_layer)  # 1 to ~0
        ddd = torch.ones(1, 1, cfg.n_embd)
        for i in range(cfg.n_embd):
            ddd[0, 0, i] = i / cfg.n_embd
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0 ** 4))

        # 参考写的是 args.n_embd * 4；这里用 T1 的 dim_ffn（只缩维度不砍组件）
        self.key = nn.Linear(cfg.n_embd, cfg.dim_ffn, bias=False)
        self.value = nn.Linear(cfg.dim_ffn, cfg.n_embd, bias=False)

        self.key.weight.data.uniform_(-0.5 / (cfg.n_embd ** 0.5), 0.5 / (cfg.n_embd ** 0.5))
        self.value.weight.data.zero_()

        self.fq_act = _fq()
        self.fq_param = _fq()
        del ddd

    def forward(self, x, ffn_prev=None):
        xx = self.fq_act(token_shift(self.time_shift, x, ffn_prev))
        k = self.fq_act(torch.addcmul(x, xx, self.fq_param(self.x_k)))
        k = self.fq_act(torch.relu(self.key(k)) ** 2)
        return self.value(k)


class Block(nn.Module):
    """一层：ln0（仅第 0 层）-> att 残差 -> ffn 残差（移植参考的 `Block`）。

    输入：forward(x, v_first, att_prev=None, ffn_prev=None, wkv_state=None)，含义同 Tmix。
    输出：(x, v_first, wkv_state_out, att_next, ffn_next)；att_next / ffn_next 是
          ln1(x) / ln2(x) 的最后一个 token，供逐 token 推理续接 time-shift。
    """

    def __init__(self, cfg: NanoConfig, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(cfg.n_embd)

        self.att = RWKV_Tmix_x070(cfg, layer_id)
        self.ffn = RWKV_CMix_x070(cfg, layer_id)

        self.fq_act = _fq()
        self.fq_param = _fq()

    def _norm(self, ln, x):
        """输入：LayerNorm 模块、x；输出：定点化的 LayerNorm 结果。

        预期行为：weight/bias 走 int8 定点常量、输出再量化回 int8。
        """
        return self.fq_act(F.layer_norm(
            x, (self.cfg.n_embd,),
            self.fq_param(ln.weight), self.fq_param(ln.bias), ln.eps))

    def forward(self, x, v_first, att_prev=None, ffn_prev=None, wkv_state=None):
        if self.layer_id == 0:
            x = self._norm(self.ln0, x)

        h1 = self._norm(self.ln1, x)
        attn_out, v_first, wkv_state_out = self.att(h1, v_first, att_prev, wkv_state)
        x = self.fq_act(x + attn_out)

        h2 = self._norm(self.ln2, x)
        x = self.fq_act(x + self.ffn(h2, ffn_prev))

        # clone：切片是视图，会把整个 (B,T,C) 激活钉在显存里
        return x, v_first, wkv_state_out, h1[:, -1].clone(), h2[:, -1].clone()


class RWKVState:
    """逐 token 推理的跨步状态（参考没有，是本项目为推理加的部分）。

    输入：由 `RWKVState.zeros` 构造，或由 `NanoRWKV.forward` 返回。
    输出：可直接传回 `NanoRWKV.forward` 的 state。
    内容：每层的 time-shift 上一 token（att / ffn 各一份，dtype 跟随隐藏状态）与
          wkv 状态 (B,H,N,N) fp32（部署时存 int32）。
    """

    __slots__ = ("att_prev", "ffn_prev", "wkv_state")

    def __init__(self, att_prev, ffn_prev, wkv_state):
        self.att_prev = att_prev
        self.ffn_prev = ffn_prev
        self.wkv_state = wkv_state

    @classmethod
    def zeros(cls, batch, cfg: NanoConfig, device, dtype=torch.float32):
        """输入：batch 大小、配置、设备、dtype；输出：全零的初始状态。"""
        H, N = cfg.dim_att // cfg.head_size, cfg.head_size
        prev = lambda: torch.zeros(batch, cfg.n_embd, device=device, dtype=dtype)  # noqa: E731
        return cls(
            att_prev=[prev() for _ in range(cfg.n_layer)],
            ffn_prev=[prev() for _ in range(cfg.n_layer)],
            wkv_state=[
                torch.zeros(batch, H, N, N, device=device, dtype=torch.float32)
                for _ in range(cfg.n_layer)
            ],
        )

    def detach(self):
        """输入：无；输出：无。原地切断所有状态张量的梯度（BPTT 截断点）。"""
        for i in range(len(self.att_prev)):
            self.att_prev[i] = self.att_prev[i].detach()
            self.ffn_prev[i] = self.ffn_prev[i].detach()
        for i in range(len(self.wkv_state)):
            self.wkv_state[i] = self.wkv_state[i].detach()
        return self


class NanoRWKV(nn.Module):
    """移植参考的 `RWKV`（去掉 lightning / deepspeed），加 state 续接。

    输入：forward(idx, state=None)；idx (B,T) 的字节 id，state 为 None 时从零开始。
    输出：(logits (B,T,V), state_out)。
    """

    def __init__(self, cfg: NanoConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.n_embd % 32 == 0
        assert cfg.dim_att % 32 == 0
        assert cfg.dim_ffn % 32 == 0

        self.emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.ln_out = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.fq_act = _fq()
        self.fq_param = _fq()
        self.apply_init()

    @torch.no_grad()
    def apply_init(self):
        """移植参考的 `RWKV.generate_init_weight`（去掉打印与 fp16/bf16 转换）。

        输入：无。输出：无（原地写权重）。
        预期行为：`ln_*` / `time_*` / 非 `.weight` 的参数保持原值（`ln_x.weight` 例外，
                  按层缩放 (1+layer_id)/n_layer 的 0.7 次方）；`emb.weight` 用 ±1e-4 的
                  均匀分布（初始 logits≈0）；`head.weight` 用正交 gain =
                  0.5*sqrt(V/C)（V>C 时）；其余 `.weight` 按 zero / 0.1 / 1.0 三种 scale
                  初始化（`.att.output.` 与 `.ffn.value.` 置零）。
        """
        sd = self.state_dict()
        for n in sd:
            p = sd[n]
            scale = 1.0
            if ("ln_" in n or ".ln" in n or "time_" in n or "_mask" in n
                    or "pos_emb" in n or ".mask." in n or n.endswith("_w")
                    or n.endswith("_w1") or n.endswith("_w2") or n.endswith("_bias")
                    or (".weight" not in n)):
                if "ln_x.weight" in n:
                    layer_scale = (1 + int(n.split(".")[1])) / self.cfg.n_layer
                    p.copy_((p * 0.0) + (layer_scale ** 0.7))
            elif n == "emb.weight":
                scale = -1e-4
                nn.init.uniform_(p, a=scale, b=-scale)
            elif n == "head.weight":
                if self.cfg.vocab_size > self.cfg.n_embd:
                    scale = 0.5 * math.sqrt(self.cfg.vocab_size / self.cfg.n_embd)
                else:
                    scale = 0.5
                nn.init.orthogonal_(p, gain=scale)
            else:
                assert n.endswith(".weight"), n
                zero = [
                    ".att.output.",
                    ".ffn.value.",
                    ".ffn.receptance.",
                    ".ffnPre.value.",
                    ".ffnPre.receptance.",
                    "head_q.",
                    ".oo.",
                    ".rr.",
                ]
                for kk in zero:
                    if kk in n:
                        scale = 0
                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1
                if scale == 0:
                    nn.init.zeros_(p)
                elif scale < 0:
                    nn.init.uniform_(p, a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(p, gain=scale)

    def forward(self, idx, state=None):
        B, T = idx.shape
        # 参考用 assert；这里用 ValueError 是为了能被调用方接住（数值语义不变）
        if T > self.cfg.ctx_len:
            raise ValueError(f"序列长度 {T} 超过 ctx_len {self.cfg.ctx_len}")

        x = self.emb(idx)

        # 参考原样：先分配一块空的 v_first，第 0 层一定会覆盖它
        v_first = torch.empty_like(x)
        fresh = state is None
        if fresh:
            state = RWKVState.zeros(B, self.cfg, x.device, x.dtype)

        att_next, ffn_next, wkv_next = [], [], []
        for i, block in enumerate(self.blocks):
            # 全零 state 与 None 等价；传 None 时 wkv7 才能走 CUDA kernel 快路径
            # （kernel 只支持从零状态开始）。外部传进来的非空 state 仍走分块实现。
            wkv_in = None if fresh else state.wkv_state[i]
            x, v_first, wkv_out, a_next, f_next = block(
                x, v_first, state.att_prev[i], state.ffn_prev[i], wkv_in
            )
            att_next.append(a_next)
            ffn_next.append(f_next)
            wkv_next.append(wkv_out)

        x = self.fq_act(F.layer_norm(
            x, (self.cfg.n_embd,),
            self.fq_param(self.ln_out.weight), self.fq_param(self.ln_out.bias),
            self.ln_out.eps))
        # head 走 QATLinear，logits 存 int8
        x = self.fq_act(self.head(x))
        return x, RWKVState(att_next, ffn_next, wkv_next)

    def parameter_count(self) -> int:
        """输入：无；输出：模型参数量（= int8 权重字节数）。"""
        return sum(p.numel() for p in self.parameters())
