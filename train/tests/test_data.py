"""数据管线测试：SFT 的补位/mask 语义、预训练窗口切分。"""

import json

import numpy as np
import torch

from src.data import PretrainDataset, SFTDataset, build_pretrain_stream, build_sft_arrays
from src.tokenizer import ETX_ID, PAD_ID


def _write_jsonl(path, texts):
    """输入：目标路径、文本列表；输出：无（写 jsonl）。"""
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


def test_build_sft_arrays_masks_prompt_and_last_position(tmp_path):
    """模板前缀与末尾 <ETX> 不参与 loss；回答段（含第 0 字节）必须全部参与。"""
    texts = ["user:你好\nbot:喵", "user:ab\nbot:cd"]
    src = tmp_path / "s.jsonl"
    _write_jsonl(src, texts)
    out = tmp_path / "built"
    meta = build_sft_arrays(str(src), str(out), ctx_len=32)

    assert meta["kept"] == 2 and meta["dropped_over_ctx"] == 0
    assert meta["no_template"] == 0
    x = np.load(out / "sft_x.npy")
    y = np.load(out / "sft_y.npy")
    mask = np.load(out / "sft_mask.npy")

    for i, t in enumerate(texts):
        # 打包时末尾补一个 <ETX>，所以落盘长度是样本字节数 + 1
        raw = t.encode("utf-8")
        data = np.frombuffer((t + chr(ETX_ID)).encode("utf-8"), dtype=np.uint8)
        n = len(data)
        # 回答段从 `\nbot:` 之后开始；它之前（含 `bot:` 本身）整段不参与 loss
        start = raw.index(b"\nbot:") + len(b"\nbot:")
        assert n == len(raw) + 1 and int(data[-1]) == ETX_ID
        # 位置 start-1 负责预测答案第 0 字节，必须参与 loss；再往前才是纯前缀
        assert mask[i, :start - 1].sum() == 0
        assert mask[i, start - 1] == 1
        assert int(mask[i, start - 1:n - 1].sum()) == n - 1 - (start - 1)
        assert mask[i, n - 1] == 0
        # 有效位上 y 就是下一个字节；倒数第二位学到的正是「这里该吐 <ETX>」
        assert np.array_equal(x[i, :n], data)
        assert np.array_equal(y[i, :n - 1], data[1:])
        assert int(y[i, n - 2]) == ETX_ID
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
    """字节流按 jsonl 顺序拼接、每篇文档后补一个 <ETX>；窗口的 y 是 x 右移一位。"""
    src = tmp_path / "p.jsonl"
    _write_jsonl(src, ["abc", "de"])
    bin_path = tmp_path / "p.bin"
    total = build_pretrain_stream(str(src), str(bin_path))
    # 每篇文档后补一个 <ETX>：3 + 1 + 2 + 1
    assert total == 7
    assert bin_path.read_bytes() == b"abc\x03de\x03"
    # 幂等：已存在时直接返回大小，不重写
    assert build_pretrain_stream(str(src), str(bin_path)) == 7

    ds = PretrainDataset(str(bin_path), ctx_len=2)
    x, y = ds[0]
    assert x.tolist() == [ord("a"), ord("b")]
    assert y.tolist() == [ord("b"), ord("c")]


def test_pad_id_is_zero():
    """PAD_ID 必须能安全当补位字节用。"""
    assert PAD_ID == 0
