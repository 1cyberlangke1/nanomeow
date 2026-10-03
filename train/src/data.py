"""数据管线：把清洗后的 jsonl 变成可以直接喂模型的字节窗口。

预训练：`pretrain_clean.jsonl` 的正文按顺序拼成一条 token 流（binidx 的做法），
**每篇文档后补一个 `<ETX>`（0x03）**标出文档边界，再按 `ctx_len` 切窗口，
**不做长度剔除**。流落到 `built/pretrain.bin`，用 np.memmap 随机读，
不把 1.2GB 全塞进内存。

SFT：`nana_clean.jsonl` 每条样本本身就是 `user:<内容>\nbot:<内容>`，打包时在末尾补
一个 `<ETX>`（0x03）作为序列结束；整条样本（含 `<ETX>`）UTF-8 字节数 > `ctx_len`
的丢弃，剩下的右补 `PAD_ID` 并给出 mask。mask 只对**回答段**（`\nbot:` 之后到
`<ETX>` 之前）置 1：模板前缀 `user:<内容>\nbot:` 是推理时由调用方给定的，模型不该
花容量去预测它；补位也不能参与 loss，否则模型会学着吐 NUL 字节。

`<ETX>` 只在打包这一步加，清洗产物里不含任何控制字节——正文里已经剔干净了，
0x03 才只可能是结构标记。

两类产物都在 `train/dataset/built/` 下，属于中间产物（已 .gitignore）。
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import ETX_ID, PAD_ID, encode

# SFT 模板 `user:<内容>\nbot:<内容>` 的分隔标记：它之前（含 `bot:` 本身）是调用方
# 给定的前缀，不参与 loss。
BOT_MARK = b"\nbot:"


def answer_start(data):
    """输入：SFT 样本的字节序列（list[int] 或 bytes）；输出：回答段起始下标，或 None。

    预期行为：定位模板分隔标记 `\nbot:`，返回它之后的位置；样本里没有这个标记时
              返回 None，由调用方决定怎么处理。
    """
    pos = bytes(data).find(BOT_MARK)
    return None if pos < 0 else pos + len(BOT_MARK)


def _iter_texts(jsonl_path):
    """输入：jsonl 路径；输出：逐行的 `text` 字段（生成器，不占内存）。"""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text")
            if text:
                yield text


def build_pretrain_stream(jsonl_path, bin_path):
    """把 jsonl 的正文拼成一条 uint8 字节流落盘。

    输入：源 jsonl、目标 .bin 路径。
    输出：字节总数（含每篇文档后的 `<ETX>`）。目标文件已存在且非空时直接返回其大小
          （幂等，不重复跑）。
    """
    if os.path.exists(bin_path) and os.path.getsize(bin_path) > 0:
        return os.path.getsize(bin_path)

    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    etx = bytes([ETX_ID])
    total = 0
    tmp_path = bin_path + ".part"
    with open(tmp_path, "wb") as out:
        for text in _iter_texts(jsonl_path):
            data = text.encode("utf-8")
            out.write(data)
            out.write(etx)
            total += len(data) + 1
    os.replace(tmp_path, bin_path)
    return total


class PretrainDataset(Dataset):
    """预训练窗口数据集。

    输入：字节流 .bin 路径、窗口长度 ctx_len。
    输出：__getitem__ 返回 (x (T,) int64, y (T,) int64)，y 是 x 右移一位。
    """

    def __init__(self, bin_path, ctx_len):
        self.ctx_len = ctx_len
        self.stream = np.memmap(bin_path, dtype=np.uint8, mode="r")
        self.n_window = max(1, (len(self.stream) - 1) // ctx_len)

    def __len__(self):
        return self.n_window

    def __getitem__(self, i):
        start = i * self.ctx_len
        window = self.stream[start:start + self.ctx_len + 1].astype(np.int64)
        x = torch.from_numpy(window[:-1])
        y = torch.from_numpy(window[1:].copy())
        return x, y


def build_sft_arrays(jsonl_path, out_dir, ctx_len):
    """把 SFT 样本编码、末尾补 `<ETX>`、卡长度、右补 PAD，落成三份 .npy。

    输入：源 jsonl、输出目录、窗口长度。
    输出：dict，含保留/丢弃条数、缺少模板分隔标记的条数与窗口长度。产物已存在时
          直接读回来（幂等）。
    预期行为：mask 只对回答段（`\nbot:` 之后到 `<ETX>` 之前）置 1，模板前缀与补位
              都是 0；找不到分隔标记的样本退回「整条都是内容」并在 meta 里计数。
    """
    x_path = os.path.join(out_dir, "sft_x.npy")
    meta_path = os.path.join(out_dir, "sft_meta.json")
    if os.path.exists(x_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    kept, dropped = [], 0
    for text in _iter_texts(jsonl_path):
        data = encode(text) + [ETX_ID]
        if len(data) > ctx_len:
            dropped += 1
            continue
        kept.append(data)

    n = len(kept)
    x = np.full((n, ctx_len), PAD_ID, dtype=np.uint8)
    mask = np.zeros((n, ctx_len), dtype=np.uint8)
    no_template = 0
    for i, data in enumerate(kept):
        x[i, :len(data)] = data
        # 最后一位是 <ETX>，它没有「下一个 token」（y 那里是补的 PAD_ID），必须排除在
        # loss 之外；倒数第二位正好学到「这里该吐 <ETX> 了」，序列结束信号就是这么训出来的。
        # 模板前缀 `user:<内容>\nbot:`（含 `bot:`）同样排除：推理时它由调用方给定，
        # 模型只需要学「给定前缀之后该吐什么」。
        start = answer_start(data)
        if start is None:
            no_template += 1
            start = 0
        mask[i, start:max(start, len(data) - 1)] = 1
    y = np.full((n, ctx_len), PAD_ID, dtype=np.uint8)
    y[:, :-1] = x[:, 1:]

    np.save(x_path, x)
    np.save(os.path.join(out_dir, "sft_y.npy"), y)
    np.save(os.path.join(out_dir, "sft_mask.npy"), mask)
    meta = {"kept": n, "dropped_over_ctx": dropped, "no_template": no_template,
            "ctx_len": ctx_len}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


class SFTDataset(Dataset):
    """SFT 数据集（已补齐、已带 mask）。

    输入：build_sft_arrays 的输出目录。
    输出：__getitem__ 返回 (x (T,) int64, y (T,) int64, mask (T,) float32)。
    """

    def __init__(self, out_dir):
        self.x = np.load(os.path.join(out_dir, "sft_x.npy"))
        self.y = np.load(os.path.join(out_dir, "sft_y.npy"))
        self.mask = np.load(os.path.join(out_dir, "sft_mask.npy"))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.x[i].astype(np.int64)),
            torch.from_numpy(self.y[i].astype(np.int64)),
            torch.from_numpy(self.mask[i].astype(np.float32)),
        )
