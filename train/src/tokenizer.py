"""字节级 tokenizer：编码 = UTF-8 字节流，解码 = 字节流。

不做 BPE、不训分词器（PLAN.md §5.1）。V = 256，任意文本都能编码，不存在 OOV，
MCU 端零分词逻辑——串口收到的字节就是 token。

生成时模型一次只吐一个字节，单个字节可能只是某个 UTF-8 字符的前半截，
直接解码会得到 U+FFFD 乱码。`UTF8StreamDecoder` 负责按字符边界吐字：
完整字符立刻输出，半截序列留在缓冲里等后续字节，非法字节直接丢掉。
"""

from typing import Iterable, List

VOCAB_SIZE = 256
PAD_ID = 0                 # SFT 批次补齐用；训练时按位置屏蔽，不参与 loss
STOP_MARKER = "user:"      # 生成到下一个 user: 就停（PLAN.md §5.2）


def encode(text: str) -> List[int]:
    """输入：任意 str；输出：UTF-8 字节值列表，每个元素落在 0..255。"""
    return list(text.encode("utf-8"))


def decode(ids: Iterable[int], errors: str = "strict") -> str:
    """输入：0..255 的字节序列；输出：解码后的 str。

    errors="strict" 时遇到非法字节会抛 UnicodeDecodeError（用于自检）；
    统计或展示场景可以传 errors="replace"。
    """
    data = bytes(int(i) & 0xFF for i in ids)
    return data.decode("utf-8", errors=errors)


class UTF8StreamDecoder:
    """增量 UTF-8 解码器：保证输出的永远是完整字符。

    输入：逐个 feed 的字节（0..255）。
    输出：每次 push 返回本次新解码出的完整文本；半截序列留在内部缓冲，
          非法字节直接丢弃（不会变成 U+FFFD）。
    """

    def __init__(self) -> None:
        self._buf: bytearray = bytearray()
        self.dropped_bytes: int = 0

    @property
    def pending_bytes(self) -> int:
        """当前还压在缓冲里、尚未凑成完整字符的字节数。"""
        return len(self._buf)

    def push(self, byte: int) -> str:
        """输入一个字节；输出这次能解出的完整文本（可能为空串）。"""
        self._buf.append(int(byte) & 0xFF)
        out: List[str] = []
        while self._buf:
            try:
                out.append(bytes(self._buf).decode("utf-8"))
            except UnicodeDecodeError as exc:
                if exc.reason == "unexpected end of data":
                    # 尾部的半截序列：先把前面完整的部分吐出去，剩下的继续等
                    if exc.start > 0:
                        out.append(bytes(self._buf[:exc.start]).decode("utf-8"))
                        del self._buf[:exc.start]
                    break
                # 非法字节：丢掉它继续，避免整条流被一个坏字节卡死
                del self._buf[exc.start]
                self.dropped_bytes += 1
            else:
                self._buf.clear()
        return "".join(out)

    def push_many(self, ids: Iterable[int]) -> str:
        """输入一串字节；输出这些字节能解出的完整文本。"""
        return "".join(self.push(i) for i in ids)

    def flush(self) -> str:
        """输入：无；输出：固定为空串——结尾没凑成字符的半截字节被丢弃。

        不返回 U+FFFD：宁可少吐一个字，也不吐乱码。
        """
        self.dropped_bytes += len(self._buf)
        self._buf.clear()
        return ""


def find_stop(text: str) -> int:
    """输入：已解码文本；输出：`user:` 出现的位置，没有则返回 -1。

    只在完整字符边界上查找，所以不会把半截 UTF-8 序列误判成停止标记。
    """
    return text.find(STOP_MARKER)