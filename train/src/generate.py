"""推理生成：逐字节自回归 + 增量 UTF-8 解码，保证不吐乱码。

停止条件：模型吐出 `<ETX>`（0x03，End of Text）就停——它是训练里唯一的结构标记，
也是序列结束信号；到达 `max_new_tokens` 也停。

「不乱码」的落点：模型一次只吐一个字节，一个 UTF-8 字符可能是 2~4 个字节，
所以**不能**逐字节解码。全部字节先过 `UTF8StreamDecoder`，只有凑成完整字符
才进输出；结尾没凑齐的半截字节直接丢，宁可少吐一个字也不吐 U+FFFD。

用法：
    python -m src.generate --ckpt out/sft/sft.pth --prompt "你好呀"
"""

import argparse

import torch

from .config import NanoConfig
from .model import NanoRWKV
from .tokenizer import ETX_ID, UTF8StreamDecoder, encode


def build_prompt(user_text):
    """输入：用户内容；输出：`user:<内容>\\nbot:` —— 冒号后没有空格。"""
    return f"user:{user_text}\nbot:"


def stream_decode(byte_iter, max_bytes=None, stop_byte=ETX_ID):
    """把字节流按完整字符吐出来，命中停止字节就截断。

    输入：字节迭代器、字节上限（None=不设）、停止字节（默认 `<ETX>` 0x03）。
    输出：(text, hit)——text 是停止字节之前的完整字符，hit 表示是否命中停止字节。
    预期行为：停止字节本身不输出；半截 UTF-8 序列留在解码器缓冲里（不输出）；
              非法字节直接丢弃。
    """
    dec = UTF8StreamDecoder()
    text = ""
    for i, byte in enumerate(byte_iter):
        if max_bytes is not None and i >= max_bytes:
            break
        if (int(byte) & 0xFF) == stop_byte:
            return text, True
        text += dec.push(byte)
    return text, False


def _sample(logits, temperature, top_k):
    """输入：最后一位的 logits (V,)、温度、top_k；输出：一个字节 id。

    temperature <= 0 时走贪心（可复现）；否则按 softmax 采样，top_k > 0 时先截断。
    """
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_k and top_k < probs.numel():
        values, _ = torch.topk(probs, top_k)
        probs = torch.where(probs >= values[-1], probs, torch.zeros_like(probs))
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate(model, prompt, max_new_tokens=256, temperature=0.0, top_k=0):
    """输入：模型、prompt 字符串、生成上限、采样参数。

    输出：生成出来的文本（不含停止字节 `<ETX>`）。
    预期行为：prompt 按 ctx_len 分块喂入并把 state 续接过去；之后逐字节生成，
              每生成一个字节就过一次增量 UTF-8 解码；命中 `<ETX>` 立刻停。
    """
    model.eval()
    device = next(model.parameters()).device
    ctx = model.cfg.ctx_len

    def byte_stream():
        state = None
        logits = None
        ids = encode(prompt)
        for start in range(0, len(ids), ctx):
            piece = ids[start:start + ctx]
            x = torch.tensor([piece], dtype=torch.long, device=device)
            logits, state = model(x, state)
        if logits is None:  # 空 prompt：用一个占位字节起头
            x = torch.zeros(1, 1, dtype=torch.long, device=device)
            logits, state = model(x, state)
        for _ in range(max_new_tokens):
            nxt = _sample(logits[0, -1], temperature, top_k)
            yield nxt
            x = torch.tensor([[nxt]], dtype=torch.long, device=device)
            logits, state = model(x, state)

    text, _ = stream_decode(byte_stream(), max_new_tokens)
    return text


def load_model(ckpt_path, device="cpu"):
    """输入：checkpoint 路径、设备；输出：加载好的 NanoRWKV（eval 模式）。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = NanoConfig(**ckpt["nano_cfg"])
    model = NanoRWKV(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    """输入：命令行；输出：把生成结果打到 stdout。"""
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="你好呀")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model = load_model(args.ckpt, args.device)
    prompt = build_prompt(args.prompt)
    print(f"[prompt] {prompt!r}")
    text = generate(model, prompt, args.max_new_tokens, args.temperature, args.top_k)
    print(f"[reply] {text!r}")
    print("[reply/raw]")
    print(text)


if __name__ == "__main__":
    main()