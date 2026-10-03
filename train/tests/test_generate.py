"""生成路径测试：增量 UTF-8 解码、停止标记截断、端到端不乱码。"""

import torch

from src.config import NanoConfig
from src.generate import build_prompt, generate, stream_decode
from src.model import NanoRWKV
from src.tokenizer import UTF8StreamDecoder, encode


class _StubModel(torch.nn.Module):
    """假模型：按脚本逐字节吐字，只用来测生成循环的解码/停止逻辑。

    输入：forward(x, state) 与真模型同签名。
    输出：一个把 scripted[i] 拉满的 logits（贪心必然选中它），i 每次调用 +1，
          越界后重复最后一个字节（模拟「一直吐同一个字节」）。
    """

    def __init__(self, scripted, ctx_len=1024):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.cfg = NanoConfig(ctx_len=ctx_len)
        self.scripted = list(scripted)
        self.i = 0

    def forward(self, x, state=None):
        b = self.scripted[min(self.i, len(self.scripted) - 1)]
        self.i += 1
        logits = torch.full((x.shape[0], x.shape[1], 256), -1e4)
        logits[..., b] = 0.0
        return logits, state


def test_build_prompt_has_no_space_after_colon():
    """模板必须是 `user:<内容>\\nbot:`，冒号后没有空格。"""
    assert build_prompt("你好") == "user:你好\nbot:"


def test_stream_decode_holds_partial_utf8():
    """一个汉字拆成 3 个字节喂进去，前两次必须什么都不吐，第三次才出字。"""
    data = encode("喵")
    assert len(data) == 3
    dec = UTF8StreamDecoder()
    assert dec.push(data[0]) == ""
    assert dec.push(data[1]) == ""
    assert dec.push(data[2]) == "喵"
    assert dec.pending_bytes == 0
    # 结尾只喂半截：flush 必须丢干净，不吐 U+FFFD
    dec2 = UTF8StreamDecoder()
    assert dec2.push_many(data[:2]) == ""
    assert dec2.flush() == ""
    assert dec2.dropped_bytes == 2


def test_stream_decode_drops_invalid_bytes():
    """非法字节必须被丢掉，绝不能变成 U+FFFD。"""
    text, _ = stream_decode(iter([0xFF, 0xFE, ord("a"), 0xFF]), stop_marker="\u0000")
    assert text == "a"
    assert "\ufffd" not in text


def test_stream_decode_truncates_at_marker():
    """命中停止标记后，标记本身与后面的字节都要丢。"""
    text, hit = stream_decode(iter(encode("喵喵user:尾巴")), stop_marker="user:")
    assert hit is True
    assert text == "喵喵"


def test_stream_decode_respects_max_bytes():
    """字节上限生效，且截断处不吐半截字符。"""
    text, hit = stream_decode(iter(encode("喵喵")), stop_marker="\u0000", max_bytes=4)
    assert hit is False
    assert text == "喵"


def test_generate_stops_at_user_marker():
    """模型吐到 user: 必须立刻停，返回标记之前的完整字符。"""
    stub = _StubModel(encode("喵喵user:后面还有一堆"))
    out = generate(stub, build_prompt("在吗"), max_new_tokens=64)
    assert out == "喵喵"


def test_generate_never_emits_replacement_char():
    """逐字节流里混着半截字符时，输出必须是合法 UTF-8 且没有 U+FFFD。"""
    stub = _StubModel(encode("喵"))  # 之后一直重复最后一字节（非法续字节）
    out = generate(stub, build_prompt("在吗"), max_new_tokens=8)
    assert out == "喵"
    assert "\ufffd" not in out
    out.encode("utf-8")  # 不抛异常 = 合法 UTF-8


def test_generate_with_real_model_is_clean_utf8():
    """真模型（未训练）跑一遍：输出必须能严格解码，且长度不超上限。"""
    torch.manual_seed(0)
    model = NanoRWKV(NanoConfig())
    out = generate(model, build_prompt("你好"), max_new_tokens=48)
    out.encode("utf-8").decode("utf-8", errors="strict")
    assert "\ufffd" not in out
    assert len(out.encode("utf-8")) <= 48


def test_generate_chunks_long_prompt():
    """prompt 超过 ctx_len 时要分块喂入而不是直接崩。"""
    model = NanoRWKV(NanoConfig(ctx_len=16))
    out = generate(model, build_prompt("这是一段明显超过十六个字节的用户输入"), max_new_tokens=8)
    out.encode("utf-8").decode("utf-8", errors="strict")