"""nanomeow 的模型与训练超参。

T1 是 PLAN.md 的基线配置：V=256 / L=2 / C=32 / head_size=8 / n_head=4 /
dim_ffn=64；低秩 rank 不单独设，按原版公式由 C 与 head_size 推出。
"""

from dataclasses import dataclass


@dataclass
class NanoConfig:
    """模型结构配置。

    输入：无，各字段都有 T1 的默认值。
    输出：可直接构造 NanoRWKV 的配置对象；n_head 由 n_embd / head_size 推出。
    """

    vocab_size: int = 256      # 字节 tokenizer：0..255
    n_layer: int = 2
    n_embd: int = 32
    dim_att: int = 32          # 参考的 args.dim_att；参考里必须等于 n_embd
    head_size: int = 8
    dim_ffn: int = 64
    ctx_len: int = 512
    wkv_chunk: int = 16        # 分块 wkv7 的块长，对齐参考实现的 CHUNK_LEN

    def __post_init__(self):
        # 参考 Tmix 用 x.view(B, T, n_head, -1) 切通道，所以 dim_att 必须等于 n_embd
        if self.dim_att != self.n_embd:
            raise ValueError("dim_att 必须等于 n_embd（参考实现的隐含前提）")
        if self.dim_att % self.head_size:
            raise ValueError("dim_att 必须能被 head_size 整除")
        if self.head_size > self.wkv_chunk:
            raise ValueError("head_size 不能大于 wkv 块长")

    @property
    def n_head(self) -> int:
        """注意力头数 = dim_att / head_size（参考的 args.dim_att // head_size）。"""
        return self.dim_att // self.head_size

    @property
    def lora_rank(self) -> int:
        """四组低秩对的 rank（原版公式；逐组系数略有不同，见 Tmix）。"""
        factor = self.head_size / 64
        return max(32, int(round((2.5 * (self.n_embd ** 0.5)) * factor / 32) * 32))

    def param_count(self) -> int:
        """按模型结构算参数量，用于和实测值对账。

        单层 = 4*C*C（r/k/v/output）
             + 11*C（x_r..x_g + Cmix.x_k + ln1/ln2 的 weight+bias）
             + 8*C（w0,a0,v0,k_k,k_a,r_k + ln_x 的 weight+bias）
             + 2*C*(D_decay + D_aaa + D_mv + D_gate)（四组低秩对）
             + 2*C*dim_ffn（Cmix key/value）
        全局 = 2*C（ln_out）+ V*C（emb）+ V*C（head，独立不共享）+ 2*C（第 0 层的 ln0）
        """
        c, f, v, l = self.n_embd, self.dim_ffn, self.vocab_size, self.n_layer
        factor = self.head_size / 64
        d_decay = max(32, int(round((2.5 * (c ** 0.5)) * factor / 32) * 32))
        d_aaa = max(32, int(round((2.5 * (c ** 0.5)) * factor / 32) * 32))
        d_mv = max(32, int(round((1.7 * (c ** 0.5)) * factor / 32) * 32))
        d_gate = max(32, int(round((5 * (c ** 0.5)) / 32) * 32))
        per_layer = (4 * c * c + 11 * c + 8 * c
                     + 2 * c * (d_decay + d_aaa + d_mv + d_gate)
                     + 2 * c * f)
        return l * per_layer + 2 * c + v * c + v * c + 2 * c


@dataclass
class TrainConfig:
    """训练超参。

    输入：无，各字段有默认值。
    输出：训练入口直接使用的配置对象。
    """

    batch_size: int = 64
    lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_steps: int = 50
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_steps: int = 2000
    log_every: int = 20
    save_every: int = 500
    num_workers: int = 4
    seed: int = 1234
    compile: bool = True
    precision: str = "bf16"
