"""让 tests/ 能 `from src... import`：把 train/ 放进 sys.path。

另外让 torch.compile 的产物落到仓库内的持久目录，跨次运行复用。
编译线程数不在这里设：inductor 在 win32 上默认 `compile_threads = 1`
（torch/_inductor/config.py::decide_compile_threads），要用多核请在环境里设
`TORCHINDUCTOR_COMPILE_THREADS`。
"""

import os

os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "compile_cache"),
)