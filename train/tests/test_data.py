"""数据管线测试：SFT 的补位/mask 语义、预训练窗口切分。"""

import json

import numpy as np
import torch

from src.data import PretrainDataset, SFTDataset, build_pretrain_stream, build_sft_arrays
from src.tokenizer import PAD_ID


def _write_jsonl(path, texts):
    """输入：目标路径、文本列表；输出：无（写 jsonl）。"""
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


def test_build_sft_arrays_masks_last_position(tmp_path):
    """样本最后一个位置的 target 是补位 PAD，必须不参与 loss。"""
    texts = ["user:你好\nbot:喵", "user:ab\nbot:cd"]
    src = tmp_path / "s.jsonl"
    _write_jsonl(src, texts)
    out = tmp_path / "built"
    meta = build_sft_arrays(str(src), str(out), ctx_len=32)

    assert meta["kept"] == 2 and meta["dropped_over_ctx"] == 0
    x = np.load(out / "sft_x.npy")
    y = np.load(out / "sft_y.npy")
    mask = np.load(out / "sft_mask.npy")

    for i, t in enumerate(texts):
        n = len(t.encode("utf-8"))
        # 有效位恰好是「除最后一位以外」的全部位置
        assert int(mask[i].sum()) == n - 1
        assert mask[i, n - 1] == 0
        # 有效位上 y 就是下一个字节
        data = np.frombuffer(t.encode("utf-8"), dtype=np.uint8)
        assert np.array_equal(x[i, :n], data)
        assert np.array_equal(y[i, :n - 1], data[1:])
        # 补位区不能有 mask
        assert mask[i, n:].sum() == 0


def test_build_sft_arrays_drops_over_ctx(tmp_path):
    """整条样本 UTF-8 字节数超过 ctx_len 的丢弃。"""
    src = tmp_path / "s.jsonl"
    _write_jsonl(src, ["短", "x" * 100])
    meta = build_sft_arrays(str(src), str(tmp_path / "built"), ctx_len=32)
    assert meta["kept"] == 1 and meta["dropped_over_ctx"] == 1


def test_sft_dataset_shapes(tmp_path):
    """SFTDataset 返回定长 (x, y, mask)，dtype 与长度正确。"""
    src = tmp_path / "s.jsonl"
    _write_jsonl(src, ["user:你好\nbot:喵"])
    out = tmp_path / "built"
    build_sft_arrays(str(src), str(out), ctx_len=32)
    ds = SFTDataset(str(out))
    x, y, mask = ds[0]
    assert x.shape == y.shape == mask.shape == (32,)
    assert x.dtype == torch.int64 and y.dtype == torch.int64 and mask.dtype == torch.float32


def test_pretrain_stream_and_windows(tmp_path):
    """字节流按 jsonl 顺序拼接；窗口的 y 是 x 右移一位。"""
    src = tmp_path / "p.jsonl"
    _write_jsonl(src, ["abc", "de"])
    bin_path = tmp_path / "p.bin"
    total = build_pretrain_stream(str(src), str(bin_path))
    assert total == 5
    # 幂等：已存在时直接返回大小，不重写
    assert build_pretrain_stream(str(src), str(bin_path)) == 5

    ds = PretrainDataset(str(bin_path), ctx_len=2)
    x, y = ds[0]
    assert x.tolist() == [ord("a"), ord("b")]
    assert y.tolist() == [ord("b"), ord("c")]


def test_pad_id_is_zero():
    """PAD_ID 必须能安全当补位字节用。"""
    assert PAD_ID == 0
