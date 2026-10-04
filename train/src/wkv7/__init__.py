"""WKV-7 递推：纯 PyTorch 分块实现 + CUDA 加速路径。

两件实现（`wkv7` 与 `wkv7_cuda`）同处一个子包，对外沿用原来的模块名，
所以调用方照旧写 `from src.wkv7 import run_wkv7, wkv7_chunked, wkv7_cuda`。
"""

from .wkv7 import CHUNK, run_wkv7, to_chunks, wkv7_chunked
from .wkv7_cuda import wkv7_cuda

__all__ = ["CHUNK", "run_wkv7", "to_chunks", "wkv7_chunked", "wkv7_cuda"]