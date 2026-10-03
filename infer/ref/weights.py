"""把训练侧的 checkpoint 变成推理侧的 QMatrix / QTensor。

输入：checkpoint 路径（.pth，含 `model` 与 `nano_cfg`）。
输出：`(NanoConfig, dict[名字] -> QMatrix | QTensor)`；QMatrix 是 2D 的 per-row 权重，
      其余是 per-tensor 的常量 / 归一化参数。
预期行为：量化口径与 `train/scripts/export_int8.py` 完全一致 —— 直接复用它的
          `collect` / `choose_scale` / `quantize` / `split_scale`，两边只留一份实现，
          所以 Python 参考引擎与 C 头文件里的权重必然同源。
"""

import importlib.util
import pathlib
import sys

import torch

from .fixed import normalize
from .int8_model import QMatrix, QTensor

TRAIN_DIR = pathlib.Path(__file__).resolve().parents[2] / "train"
# 训练侧的 `src` 包要能被 import（推理侧只在这里借用它的模型定义，不参与部署）
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))


def _export_module():
    """输入：无；输出：`train/scripts/export_int8.py` 模块对象。

    预期行为：按文件路径加载，避免依赖 `scripts` 是不是包；该脚本自己会把 `train/`
              加进 sys.path，所以 `import src.*` 能成。
    """
    if str(TRAIN_DIR) not in sys.path:
        sys.path.insert(0, str(TRAIN_DIR))
    path = TRAIN_DIR / "scripts" / "export_int8.py"
    spec = importlib.util.spec_from_file_location("nanomeow_export_int8", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scale_pair(mul, shift):
    """输入：int32 乘子、int8 移位；输出：(m, e)。预期行为：值不变，只归一到 m ∈ [2^30, 2^31)。"""
    return normalize(int(mul), int(shift))


def load_weights(ckpt_path):
    """输入：checkpoint 路径；输出：(NanoConfig, dict[名字] -> QMatrix | QTensor)。"""
    from src.config import NanoConfig
    from src.model import NanoRWKV

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = NanoConfig(**ckpt["nano_cfg"])
    model = NanoRWKV(cfg)
    model.load_state_dict(ckpt["model"])

    export = _export_module()
    out = {}
    for name, tensor, per_row in export.collect(model):
        flat = tensor.float()
        scale = export.choose_scale(flat, -1 if per_row else None)
        codes = export.quantize(flat, scale)
        mul, shift, _ = export.split_scale(scale, codes)
        if per_row:
            rows, cols = codes.shape
            out[name] = QMatrix(
                codes.reshape(-1).tolist(),
                [_scale_pair(m, s) for m, s in zip(mul.tolist(), shift.tolist())],
                rows, cols)
        else:
            out[name] = QTensor(codes.reshape(-1).tolist(), _scale_pair(mul[0], shift[0]))
    return cfg, out
