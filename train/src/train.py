"""训练入口：P2 预训练 / P3 SFT，bf16 autocast + torch.compile。

用法（在 train/ 目录下）：
    python -m src.train --stage pretrain
    python -m src.train --stage sft --init-from out/pretrain/pretrain.pth

精度口径：参数是 fp32 master，前向/反向计算走 bf16 autocast
——这就是参考实现 `--precision bf16` 的含义，不是 fp32 训练；wkv7 递推内部
额外强制 fp32（见 src/wkv7.py 的说明）。
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

from .config import NanoConfig, TrainConfig
from .data import (PretrainDataset, SFTDataset, build_pretrain_stream,
                   build_sft_arrays)
from .model import NanoRWKV
from .qat import prepare_qat

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.dirname(HERE)
DATASET_DIR = os.path.join(TRAIN_DIR, "dataset")
BUILT_DIR = os.path.join(DATASET_DIR, "built")
OUT_DIR = os.path.join(TRAIN_DIR, "out")
CACHE_DIR = os.path.join(OUT_DIR, "compile_cache")


class L2Wrap(torch.autograd.Function):
    """RWKV 官方的 loss 正则：只在每个位置的 argmax logit 上加一点点梯度，
    等价于隐式惩罚 max_logit^2，防止 bf16 下 logits 越训越大最后溢出。

    输入：forward(loss, logits)；输出：原样返回 loss。
    预期行为：backward 除了主 loss 的梯度，再额外给 argmax 位置一个
              `max_logit * 1e-4/(B*T)` 的梯度。
    """

    @staticmethod
    def forward(ctx, loss, logits):
        ctx.save_for_backward(logits)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (logits,) = ctx.saved_tensors
        maxx, ids = torch.max(logits, -1, keepdim=True)
        gy = torch.zeros_like(logits)
        gy.scatter_(-1, ids, (maxx * (1e-4 / (logits.shape[0] * logits.shape[1]))).to(logits.dtype))
        return grad_output, grad_output * gy


def parse_args():
    """输入：命令行；输出：argparse 命名空间。"""
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["pretrain", "sft"], required=True)
    p.add_argument("--ctx-len", type=int, default=512)
    p.add_argument("--wkv-chunk", type=int, default=16, help="分块 wkv7 的块长；数学等价，只影响速度与三角求解精度")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1, help="跑几遍数据；1 = 完整一遍")
    p.add_argument("--steps", type=int, default=0,
                   help="0 = 由 --epochs 按数据量算；正数 = 直接指定总步数")
    # 默认值照抄参考的 demo 脚本：demo-training-run.sh 用 lr 6e-4 -> 2e-5、
    # warmup 10、weight_decay 0.001、adam_eps 1e-18；demo-training-run-sft.sh 用
    # lr 2e-5 -> 1e-6、warmup 10（SFT 那组在命令行显式传，见 HANDOFF）。
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr", type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compile", type=int, default=1, help="1=torch.compile 包裹模型")
    p.add_argument("--qat", type=int, default=1,
                   help="1=全程 QAT（默认）；0=关掉假量化，只做对照")
    p.add_argument("--compile-mode", default="default",
                   help="default / reduce-overhead（CUDA Graph）/ max-autotune")
    p.add_argument("--init-from", default="", help="从 checkpoint 初始化（SFT 接预训练）")
    p.add_argument("--resume", default="", help="从 checkpoint 续训（含 step）")
    p.add_argument("--out", default="", help="输出目录，默认 out/<stage>")
    p.add_argument("--limit-windows", type=int, default=0,
                   help="只用前 N 个窗口（冒烟用；0=全用）")
    return p.parse_args()


def build_datasets(args, cfg):
    """输入：命令行参数与模型配置；输出：(dataset, collate 描述, 是否带 mask)。

    预训练产物是字节流 + 随机窗口；SFT 产物是补齐好的定长数组 + mask。
    """
    if args.stage == "pretrain":
        bin_path = os.path.join(BUILT_DIR, "pretrain.bin")
        total = build_pretrain_stream(os.path.join(DATASET_DIR, "pretrain_clean.jsonl"), bin_path)
        print(f"[data] 预训练字节流 {total/1e9:.3f} GB -> {bin_path}")
        ds = PretrainDataset(bin_path, args.ctx_len)
        return ds, False
    out_dir = os.path.join(BUILT_DIR, "sft")
    meta = build_sft_arrays(os.path.join(DATASET_DIR, "nana_clean.jsonl"), out_dir, args.ctx_len)
    print(f"[data] SFT 保留 {meta['kept']} 条，超长丢弃 {meta['dropped_over_ctx']} 条")
    return SFTDataset(out_dir), True


def build_optimizer(model, args):
    """三组参数（同参考实现）：w0 用 2x lr、二维 .weight 走 weight decay、其余 1x。

    输入：模型与命令行参数；输出：覆盖全部参数的 AdamW。
    """
    lr_decay, lr_1x, lr_2x = set(), set(), set()
    for n, p in model.named_parameters():
        if "att.w0" in n:
            lr_2x.add(n)
        elif p.dim() >= 2 and ".weight" in n and args.weight_decay > 0:
            lr_decay.add(n)
        else:
            lr_1x.add(n)

    table = dict(model.named_parameters())
    groups = [
        {"params": [table[n] for n in sorted(lr_1x)], "weight_decay": 0.0, "lr_scale": 1.0},
        {"params": [table[n] for n in sorted(lr_2x)], "weight_decay": 0.0, "lr_scale": 2.0},
    ]
    if lr_decay:
        groups.append({"params": [table[n] for n in sorted(lr_decay)],
                       "weight_decay": args.weight_decay, "lr_scale": 1.0})
    covered = sum(len(g["params"]) for g in groups)
    assert covered == len(list(model.parameters())), "有参数没进 optimizer"
    # betas / eps 同参考 demo 脚本（--beta1 0.9 --beta2 0.99 --adam_eps 1e-18）
    return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.99), eps=1e-18)


def lr_at(step, args):
    """输入：当前步数、参数；输出：该步的 base lr。

    预期行为：与参考 trainer 的 on_train_batch_start 逐位相同——进度按 **token** 算
              （step × ctx_len × batch），余弦在 [lr, min_lr] 之间插值，warmup 是乘在
              余弦结果上的 0.01→1.0 线性因子。
    """
    tokens = step * args.ctx_len * args.batch_size
    total_tokens = args.steps * args.ctx_len * args.batch_size
    warmup_tokens = args.warmup_steps * args.ctx_len * args.batch_size
    progress = (tokens - warmup_tokens) / max(1, total_tokens - warmup_tokens)
    progress = max(0.0, min(1.0, progress))
    final_factor = args.min_lr / args.lr
    lr = args.lr * ((0.5 + final_factor / 2)
                    + (0.5 - final_factor / 2) * math.cos(math.pi * progress))
    if step < args.warmup_steps:
        lr = lr * (0.01 + 0.99 * step / args.warmup_steps)
    return lr


def loss_fn(logits, y, mask=None):
    """输入：logits (B,T,V)、y (B,T)、可选 mask (B,T)；输出：(loss, 有效 token 数)。

    有 mask 时按有效位置取平均（补位是 PAD_ID，不能算进去）。
    """
    flat = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                           reduction="none").view(y.shape)
    if mask is None:
        return flat.mean(), y.numel()
    return (flat * mask).sum() / mask.sum().clamp_min(1.0), int(mask.sum().item())


def save_ckpt(path, model, cfg, args, step):
    """输入：目标路径、模型、配置、参数、步数；输出：无（写盘）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    torch.save({"model": model.state_dict(), "nano_cfg": asdict(cfg),
                "step": step, "stage": args.stage}, tmp)
    os.replace(tmp, path)


def main():
    """输入：无（读命令行）；输出：无（写 checkpoint 与日志）。"""
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = args.out or os.path.join(OUT_DIR, args.stage)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", CACHE_DIR)

    cfg = NanoConfig(ctx_len=args.ctx_len, wkv_chunk=args.wkv_chunk)
    model = NanoRWKV(cfg)
    print(f"[model] 参数量 {model.parameter_count():,}（int8 约 {model.parameter_count()/1024:.1f} KB）")

    # QAT 全程打开——假量化从第 0 步就进图，权重一开始就长成「取整之后好用」的形状。
    # 换模块必须在 torch.compile 之前（编译要看到假量化算子）。
    if args.qat:
        model = prepare_qat(model)
        print("[qat] 假量化插桩点已打开")

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[model] 从 {args.init_from} 初始化（step={ckpt.get('step')}）")

    device = torch.device(args.device)
    model = model.to(device)
    optimizer = build_optimizer(model, args)
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        print(f"[model] 从 {args.resume} 续训（step={start_step}）")

    dataset, use_mask = build_datasets(args, cfg)
    steps_per_epoch = max(1, len(dataset) // args.batch_size)
    if args.steps <= 0:
        args.steps = steps_per_epoch * args.epochs
    print(f"[data] 样本 {len(dataset)}，一遍 {steps_per_epoch} 步 × {args.epochs} 遍 = 总 {args.steps} 步")
    n_used = args.limit_windows or len(dataset)
    print(f"[data] 共 {len(dataset)} 个样本，本阶段用 {n_used} 个")
    indices = list(range(n_used))
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        sampler=SubsetRandomSampler(indices),
                        num_workers=0, drop_last=True)

    net = model
    if args.compile:
        if device.type == "cuda":
            # CUDA 扩展的加载（含首次编译）必须在 torch.compile 之前完成：
            # 否则 dynamo 会 trace 进 load() 的文件操作，产生图断点。
            from .wkv7_cuda import _load_ext
            _load_ext()
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        net = torch.compile(model, mode=args.compile_mode)
        print(f"[compile] torch.compile(mode={args.compile_mode}) 已启用")

    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False))
    use_cuda_graph = args.compile_mode == "reduce-overhead" and device.type == "cuda"

    step = start_step
    tokens_seen = 0
    t0 = time.time()
    log_t0 = t0
    log_loss = 0.0
    log_tokens = 0
    data_iter = iter(loader)
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        base_lr = lr_at(step, args)
        for g in optimizer.param_groups:
            g["lr"] = base_lr * g["lr_scale"]

        if use_cuda_graph:
            torch.compiler.cudagraph_mark_step_begin()

        optimizer.zero_grad(set_to_none=True)
        with autocast:
            if use_mask:
                x, y, mask = (t.to(device, non_blocking=True) for t in batch)
                logits, _ = net(x)
                loss, n_tok = loss_fn(logits, y, mask)
            else:
                x, y = (t.to(device, non_blocking=True) for t in batch)
                logits, _ = net(x)
                loss, n_tok = loss_fn(logits, y)
            loss = L2Wrap.apply(loss, logits)
        loss.backward()
        gnorm = (torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                 if args.grad_clip > 0 else None)

        # 非有限值保险丝：出现 NaN/inf 就说明权重已经不可用，继续跑只是空烧算力
        # （实测会连跳几千步、每步都算完前向后向再丢弃）。这里当场终止，并报出
        # 哪一步、loss、|g|，以及哪些参数的梯度先坏掉，供定位根因。
        step += 1
        if not torch.isfinite(loss) or (gnorm is not None and not torch.isfinite(gnorm)):
            bad_params = [n for n, p in model.named_parameters()
                          if p.grad is not None and not torch.isfinite(p.grad).all()]
            raise RuntimeError(
                f"[abort] step {step}: loss={float(loss)} "
                f"|g|={float(gnorm) if gnorm is not None else -1} "
                f"梯度非有限的参数={bad_params}，训练终止"
                f"（最后 checkpoint：{os.path.join(out_dir, args.stage)}.pth）"
            )
        optimizer.step()

        tokens_seen += n_tok
        log_loss += float(loss.item())
        log_tokens += n_tok

        if step % args.log_every == 0:
            now = time.time()
            avg = log_loss / args.log_every
            dt = now - log_t0
            print(f"[{args.stage}] step {step}/{args.steps} loss {avg:.4f} "
                  f"ppl {math.exp(min(avg, 20)):8.2f} lr {base_lr:.2e} "
                  f"|g| {float(gnorm) if gnorm is not None else -1:.3f} "
                  f"{log_tokens/dt:,.0f} tok/s 累计 {tokens_seen/1e6:.1f}M", flush=True)
            log_t0, log_loss, log_tokens = now, 0.0, 0

        if args.save_every and step % args.save_every == 0:
            path = os.path.join(out_dir, f"{args.stage}_step{step}.pth")
            save_ckpt(path, model, cfg, args, step)
            print(f"[save] {path}")

    final = os.path.join(out_dir, f"{args.stage}.pth")
    save_ckpt(final, model, cfg, args, step)
    with open(os.path.join(out_dir, f"{args.stage}_train.json"), "w", encoding="utf-8") as f:
        json.dump({"stage": args.stage, "steps": step, "tokens": tokens_seen,
                   "seconds": time.time() - t0, "final_loss": float(loss.item()),
                   "args": vars(args), "cfg": asdict(cfg)}, f, ensure_ascii=False, indent=2)
    print(f"[done] {step} 步，{tokens_seen/1e6:.1f}M token，{time.time()-t0:.1f}s -> {final}")


if __name__ == "__main__":
    main()
