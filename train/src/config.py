"""nanomeow 的模型与训练超参。

基线配置：V=256 / L=3 / C=32 / head_size=8 / n_head=4 /
dim_ffn=64；低秩 rank 见 `dim_lora`。
"""

from dataclasses import dataclass


@dataclass
class NanoConfig:
    """模型结构配置。

    输入：无，各字段都有 T1 的默认值。
    输出：可直接构造 NanoRWKV 的配置对象；n_head 由 n_embd / head_size 推出。
    """

    vocab_size: int = 256      # 字节 tokenizer：0..255
    n_layer: int = 3
    n_embd: int = 32
    dim_att: int = 32          # 参考的 args.dim_att；参考里必须等于 n_embd
    # 每头 state 是 N×N 矩阵，一个头最多存 N 个独立的 rank-1 关联；总秩是
    # n_head×N，N=8（4 头）与 N=32（1 头）都是 32，缩 N 不损失关联容量，
    # 但只有 N=8 才走得上 CUDA kernel 快路径（对齐参考的 head_size / CHUNK_LEN）。
    head_size: int = 8
    dim_ffn: int = 64
    dim_lora: int = 8          # 四组低秩对的 rank；取 8 是照参考的 D/C 比例缩维，见 lora_rank
    ctx_len: int = 512
    wkv_chunk: int = 16        # 分块 wkv7 的块长；必须 >= head_size（参考的 CHUNK_LEN 是 16）
    # 输入相关 decay：参考 x070 的 `tanh(xw @ w1) @ w2` 那一支。True = 原样移植；
    # False = decay 退回每通道静态 w0，并且**不注册** x_w / w1 / w2——它们在没有
    # 这一支时进不了任何计算，留着就是梯度恒 0 的死参数。
    dynamic_decay: bool = True

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
        """四组低秩对的 rank。

        输入：无。输出：D。
        预期行为：参考的 `max(32, ...)` 是给大模型的地板，参考实际配置
                  C=512 / head_size=64 时 D_DECAY_LORA=64，即 D/C = 1/8。
                  缩维到 C=32 后照抄地板会得到 D=32=C（低秩对退化成满秩，
                  相对容量是参考的 8 倍），所以按同一比例取 D=8。
        """
        return self.dim_lora

    def param_count(self) -> int:
        """按模型结构算参数量，用于和实测值对账。

        单层 = 4*C*C（r/k/v/output）
             + 11*C（x_r..x_g + Cmix.x_k + ln1/ln2 的 weight+bias；
                     dynamic_decay=False 时没有 x_w，是 10*C）
             + 8*C（w0,a0,v0,k_k,k_a,r_k + ln_x 的 weight+bias）
             + 2*C*4*D（四组低秩对，rank 都是 dim_lora；
                        dynamic_decay=False 时 decay 那一组不存在，是三组）
             + 2*C*dim_ffn（Cmix key/value）
        全局 = 2*C（ln_out）+ V*C（emb）+ V*C（head，独立不共享）+ 2*C（第 0 层的 ln0）
        """
        c, f, v, l = self.n_embd, self.dim_ffn, self.vocab_size, self.n_layer
        n_mix = 11 if self.dynamic_decay else 10
        n_lora = 4 if self.dynamic_decay else 3
        per_layer = (4 * c * c + n_mix * c + 8 * c
                     + 2 * c * n_lora * self.dim_lora
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
