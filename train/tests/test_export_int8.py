"""导出脚本的口径测试：导出的 (码, scale) 必须与训练侧 QAT 假量化逐位一致。"""

import pathlib
import sys

import torch

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from export_int8 import choose_scale, quantize, split_scale  # noqa: E402
from src.qat import IntxFakeQuantizer, per_row_int8, per_tensor_int8  # noqa: E402


def test_per_row_matches_qat_quantizer():
    """权重 per-row：反量化结果必须与训练侧 per_row_int8 假量化逐位相同。"""
    torch.manual_seed(0)
    w = torch.randn(17, 8) * 3
    scale = choose_scale(w, -1)
    assert torch.equal(quantize(w, scale).float() * scale,
                       IntxFakeQuantizer(per_row_int8())(w))


def test_per_tensor_matches_qat_quantizer():
    """常量 per-tensor：反量化结果必须与训练侧 per_tensor_int8 假量化逐位相同。"""
    torch.manual_seed(1)
    x = torch.randn(5, 7) * 0.3
    scale = choose_scale(x, None)
    assert torch.equal(quantize(x, scale).float() * scale,
                       IntxFakeQuantizer(per_tensor_int8())(x))


def test_scale_roundtrip_is_tight():
    """(int32 乘子, int8 移位) 表示的 scale 必须能还原到 1e-6 相对误差以内。"""
    scale = torch.tensor([1e-3, 0.25, 1.0, 7.5, 1234.5])
    codes = torch.ones(5, 4, dtype=torch.int8)
    mul, shift, zero = split_scale(scale, codes)
    back = mul.float() * torch.pow(2.0, shift.float())
    assert not zero.any()
    assert ((back - scale).abs() / scale).max().item() < 1e-6


def test_zero_rows_are_flagged_not_silently_scaled():
    """全零行的 scale 没有意义（eps），必须被标出来，且移位不能溢出 int8。"""
    w = torch.zeros(3, 4)
    w[1] = 0.5
    scale = choose_scale(w, -1)
    mul, shift, zero = split_scale(scale, quantize(w, scale))
    assert zero.tolist() == [True, False, True]
    assert int(shift.min()) >= -127 and int(shift.max()) <= 127
