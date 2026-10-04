"""把 QAT 训练出的 checkpoint 导成部署用的 int8 权重 + scale 表。

输入：--ckpt（如 train/out/sft_dyn/sft.pth）、--out（C 头文件路径）。
输出：C 头文件（int8 码 + 每行 int32 乘子 + int8 移位）与 stdout 摘要。
预期行为：口径与训练侧 QAT 完全一致——权重 per-row（输出通道）对称 int8、范围 [-128, 127]、
          映射用 SYMMETRIC_NO_CLIPPING_ERR（正负端各除自己的边界取大者）；常量张量 per-tensor。
          自检会把「导出的码 x scale」与训练侧假量化的输出逐位对比，不一致直接报错退出。

scale 的表示：value = multiplier * 2 ** shift，multiplier 归一化到 [2^30, 2^31)，
所以一条 scale 只占 int32 + int8 = 5 字节，与 PLAN 7.3 的账本一致。

用法：
    python train/scripts/export_int8.py --ckpt train/out/sft_dyn/sft.pth
"""

import argparse
import pathlib
import sys

import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))

from src.config import NanoConfig  # noqa: E402
from src.model import NanoRWKV  # noqa: E402
from src.qat import (  # noqa: E402
    INT8_MAX,
    INT8_MIN,
    IntxFakeQuantizer,
    per_row_int8,
    per_tensor_int8,
)

EPS = torch.finfo(torch.float32).smallest_normal

# 低秩对 (in, out)：存储形状是 (C, D) 或 (D, C)，模型里一律是
# `F.linear(x, p.transpose(0, 1))` 用的，所以导出布局统一取 p 的转置 = (out, in) 行主序，
# 正好也是 GEMV 要的形状。**两个方向都要转**：只转 (C, D) 那一半会让 (D, C) 那一半的
# per-row 归约维反过来（实测 w2/a2/v2/g2 的 scale 数与训练侧不符）。
LOWRANK = ("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2")


def choose_scale(x, dim):
    """输入：张量 x、归约维度 dim（None = per-tensor）；输出：scale（fp32）。

    预期行为：复刻 torchao 的 SYMMETRIC_NO_CLIPPING_ERR——
              scale = max(|min| / 128, max / 127)，再夹到 eps 以上。
    """
    if dim is None:
        mn = x.min().reshape(1)
        mx = x.max().reshape(1)
    else:
        mn = x.amin(dim=dim, keepdim=True)
        mx = x.amax(dim=dim, keepdim=True)
    smin = torch.clamp(mn, max=0.0).float() / float(INT8_MIN)
    smax = torch.clamp(mx, min=0.0).float() / float(INT8_MAX)
    return torch.clamp(torch.maximum(smin, smax), min=EPS)


def quantize(x, scale):
    """输入：张量 x、scale；输出：int8 码。

    预期行为：四舍六入五成双后夹到 [-128, 127]。必须写成 x * (1/s) 而不是 x / s：
              训练侧的 _fake_quantize_no_zero_point_ste 就是乘倒数，fp32 下两者不等价。
    """
    return torch.clamp(torch.round(x * (1.0 / scale)), INT8_MIN, INT8_MAX).to(torch.int8)


def split_scale(scale, codes):
    """输入：fp32 scale、对应的 int8 码；输出：(int32 乘子, int8 移位, 是否全零行)。

    预期行为：value = mul * 2 ** shift，乘子归一化到 [2^30, 2^31)，即 shift = floor(log2(scale)) - 30。
              移位夹在 int8 范围内；全零行（码全为 0，例如参考实现里第 0 层按设计不参与
              value-residual 的 v1）的 scale 取不到有效值，它们的贡献恒为 0，所以标记出来
              并在误差自检里单独对待，而不是让它污染 shift 的取值。
    """
    flat = scale.reshape(-1)
    zero_row = codes.reshape(flat.numel(), -1).abs().amax(dim=1) == 0
    e = torch.floor(torch.log2(flat)) - 30.0
    e = torch.clamp(e, -127.0, 127.0)
    mul = torch.round(flat / torch.pow(2.0, e)).to(torch.int64)
    mul = torch.clamp(mul, 1 << 30, (1 << 31) - 1).to(torch.int32)
    return mul, e.to(torch.int8), zero_row


def collect(model):
    """输入：模型；输出：[(名字, 导出用矩阵, 是否 per-row)]。

    预期行为：Linear / Embedding 的权重按输出通道 per-row；低秩对转成 (out, in) 后 per-row；
              其余（LayerNorm 参数、逐元素常量）per-tensor。
    """
    items = []
    for name, tensor in model.state_dict().items():
        base = name.rsplit(".", 1)[-1]
        if name in ("emb.weight", "head.weight"):
            items.append((name, tensor, True))
        elif name.endswith(".weight") and tensor.dim() == 2:
            items.append((name, tensor, True))
        elif base in LOWRANK and tensor.dim() == 2:
            items.append((name, tensor.transpose(0, 1), True))
        else:
            items.append((name, tensor.reshape(-1), False))
    return items


def export(model, out_path, source=""):
    """输入：模型、头文件路径、权重来源 checkpoint 描述；输出：统计字典（同时写头文件）。"""
    fq_row = IntxFakeQuantizer(per_row_int8())
    fq_ten = IntxFakeQuantizer(per_tensor_int8())

    exported = []
    for name, tensor, per_row in collect(model):
        t = tensor.float()
        # per-row = 对最后一维（输入维）归约，得到「每个输出通道一条」的 scale；
        # 导出矩阵已经是 (out, in) 布局，所以归约维是 -1。
        scale = choose_scale(t, -1 if per_row else None)
        codes = quantize(t, scale)
        if not torch.equal(codes.float() * scale, fq_row(t) if per_row else fq_ten(t)):
            raise SystemExit("自检失败：%s 的反量化与训练侧假量化不一致" % name)
        mul, shift, zero_row = split_scale(scale, codes)
        back = mul.float() * torch.pow(2.0, shift.float())
        rel = ((back - scale.reshape(-1)).abs() / scale.reshape(-1))[~zero_row]
        if rel.numel() and rel.max().item() > 1e-6:
            raise SystemExit("自检失败：%s 的 scale 归一化误差过大 %g"
                             % (name, rel.max().item()))
        exported.append((name, codes, mul, shift, per_row))

    zero_rows = sum(int((c.reshape(m.numel(), -1).abs().amax(dim=1) == 0).sum())
                    for _, c, m, _, _ in exported)
    n_param = sum(c.numel() for _, c, _, _, _ in exported)
    n_scale = sum(m.numel() for _, _, m, _, _ in exported)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path = out_path.with_name("model_cfg.h")
    with open(cfg_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("/* 自动生成，请勿手改：train/scripts/export_int8.py */\n")
        fh.write("#ifndef NANOMEOW_MODEL_CFG_H\n#define NANOMEOW_MODEL_CFG_H\n")
        fh.write("/* 模型结构常量：C 引擎取维度的唯一来源，改维度只需重跑导出。 */\n")
        fh.write("#define NMW_N_LAYER %d\n" % model.cfg.n_layer)
        fh.write("#define NMW_N_EMBD %d\n" % model.cfg.n_embd)
        fh.write("#define NMW_N_HEAD %d\n" % model.cfg.n_head)
        fh.write("#define NMW_HEAD_SIZE %d\n" % model.cfg.head_size)
        fh.write("#define NMW_DIM_FFN %d\n" % model.cfg.dim_ffn)
        fh.write("#define NMW_VOCAB %d\n" % model.cfg.vocab_size)
        fh.write("#define NMW_CTX_LEN %d\n" % model.cfg.ctx_len)
        fh.write("#define NMW_DYNAMIC_DECAY %d\n" % int(model.cfg.dynamic_decay))
        fh.write("#endif\n")
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("/* 自动生成，请勿手改：train/scripts/export_int8.py */\n")
        fh.write("/* 权重来源：%s */\n" % (source or "(未知)"))
        fh.write("#ifndef NANOMEOW_WEIGHTS_H\n#define NANOMEOW_WEIGHTS_H\n")
        fh.write("#include <stdint.h>\n#include \"model_cfg.h\"\n\n")
        fh.write("/* 每个张量：int8 码 + 每行 int32 乘子 + int8 移位，值 = 码 * (乘子 * 2^移位)；符号前缀 nmw_ = nano-meow weights */\n")
        fh.write("/*\n")
        fh.write(" * 本文件导出的权重（名字 = 训练侧参数名；字节数 = int8 码 / scale 表）：\n")
        for name, codes, mul, shift, per_row in exported:
            if per_row:
                kind = "per-row %d x %d" % (mul.numel(), codes.numel() // mul.numel())
            else:
                kind = "per-tensor %d" % codes.numel()
            fh.write(" *   %-32s %6d B / %5d B  (%s)\n"
                     % (name, codes.numel(), mul.numel() * 5, kind))
        fh.write(" * 合计：码 %d B（%.1f KiB）+ scale 表 %d B（%.1f KiB）= %d B（%.1f KiB）\n"
                 % (n_param, n_param / 1024.0, n_scale * 5, n_scale * 5 / 1024.0,
                    n_param + n_scale * 5, (n_param + n_scale * 5) / 1024.0))
        fh.write(" */\n")
        for name, codes, mul, shift, per_row in exported:
            sym = "nmw_" + name.replace(".", "_")
            flat = ",".join(str(v) for v in codes.reshape(-1).tolist())
            fh.write("static const int8_t %s[] = {%s};\n" % (sym, flat))
            if per_row:
                fh.write("static const int32_t %s_mul[] = {%s};\n"
                         % (sym, ",".join(str(int(v)) for v in mul.tolist())))
                fh.write("static const int8_t %s_shift[] = {%s};\n"
                         % (sym, ",".join(str(int(v)) for v in shift.tolist())))
            else:
                fh.write("static const int32_t %s_mul = %d;\n" % (sym, int(mul[0])))
                fh.write("static const int8_t %s_shift = %d;\n" % (sym, int(shift[0])))
        fh.write("\n#endif\n")

    return {"params": n_param, "scales": n_scale, "zero_rows": zero_rows,
            "header_bytes": out_path.stat().st_size}


def main():
    """输入：命令行；输出：摘要 + 头文件。"""
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=str(REPO / "infer" / "model_weights.h"))
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = NanoConfig(**ckpt["nano_cfg"])
    model = NanoRWKV(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stats = export(model, pathlib.Path(args.out), args.ckpt)
    scale_bytes = stats["scales"] * 5
    print("[cfg] n_layer=%d n_embd=%d head_size=%d dim_ffn=%d dynamic_decay=%s"
          % (cfg.n_layer, cfg.n_embd, cfg.head_size, cfg.dim_ffn, cfg.dynamic_decay))
    print("[export] int8 权重 %d 字节 = %.1f KiB" % (stats["params"], stats["params"] / 1024))
    print("[export] scale 表 %d 条 x 5B = %.1f KiB" % (stats["scales"], scale_bytes / 1024))
    print("[export] 合计 %d 字节 = %.1f KiB"
          % (stats["params"] + scale_bytes, (stats["params"] + scale_bytes) / 1024))
    print("[export] 头文件 %.0f KiB -> %s" % (stats["header_bytes"] / 1024, args.out))
    print("[export] 自检通过：%d 个张量的反量化与训练侧假量化逐位一致（其中全零行 %d 条）"
          % (len(collect(model)), stats["zero_rows"]))


if __name__ == "__main__":
    main()
