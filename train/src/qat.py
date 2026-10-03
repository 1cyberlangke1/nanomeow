"""QAT：把部署时的定点边界搬进训练（PLAN §7.1 / §7.4）。

本文件是 `tmp/ao`（pytorch/ao 的 torchao）的**移植**，不是自己写的：算子与模块
逐段照抄 torchao 的对应实现，只删掉本模型用不到的分支（非对称 / 分组 / float8 /
int4 / range learning / 静态 qparam）。

移植对照表（左 = 本文件，右 = torchao 原位置）：

| 本文件 | torchao 原位置 |
|---|---|
| `Granularity` / `PerTensor` / `PerAxis` / `PerRow` | `torchao/quantization/granularity.py` |
| `get_block_size` | `torchao/quantization/utils.py` |
| `MappingType` / `ZeroPointDomain` | `torchao/quantization/quant_primitives.py` |
| `_Round` / `_ClampSTE` | 同上 |
| `_get_and_check_qmin_qmax` | 同上 |
| `_get_reduction_params` | 同上 |
| `_quantize_affine_no_zero_point_no_dtype_cast` | 同上 |
| `_dequantize_affine_no_zero_point_no_dtype_check` | 同上 |
| `_choose_qparams_affine`（只留 SYMMETRIC 分支） | 同上 |
| `_do_fake_quantize_affine` / `_fake_quantize_affine` | 同上 |
| `_fake_quantize_per_channel_group` | `torchao/quantization/qat/utils.py` |
| `IntxFakeQuantizeConfig` | `torchao/quantization/qat/fake_quantize_config.py` |
| `FakeQuantizerBase` / `IntxFakeQuantizer` | `torchao/quantization/qat/fake_quantizer.py` |
| `FakeQuantizedLinear` | `torchao/quantization/qat/linear.py` |
| `FakeQuantizedEmbedding` | `torchao/quantization/qat/embedding.py` |

本模型的口径（PLAN §7.1，一条都不许改）：

- 权重 **int8**，per-channel（per-row）对称，范围 `[-127, 127]`（不是 `[-128, 127]`）。
- 激活 **int8**，per-tensor 动态（运行时求 max -> scale）。
- 累加 int32；对称、无 zero-point；不做分组。
"""

import enum
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

# PLAN §7.1：对称、无 zero-point，所以范围是 [-127, 127] 而不是 [-128, 127]
INT8_MIN = -127
INT8_MAX = 127

# torchao 原表里 int32 是「载体」dtype（_fake_quantize_per_channel_group 用它做中间量），
# 范围就是 int32 全域，所以这里必须留着，否则分组路径会抛 Unsupported dtype
_DTYPE_TO_QVALUE_BOUNDS = {
    torch.int8: (-128, 127),
    torch.int32: (-(2 ** 31), 2 ** 31 - 1),
}


# ---------------------------------------------------------------- 粒度
# 移植自 torchao/quantization/granularity.py（只留本模型用得到的三种）


@dataclass(frozen=True)
class Granularity:
    """量化粒度的基类（torchao.quantization.granularity.Granularity）。"""


@dataclass(frozen=True)
class PerTensor(Granularity):
    """整个张量共用一组 qparam。"""


@dataclass(frozen=True)
class PerAxis(Granularity):
    """axis 是保留维，其余维归约（torchao 的 axis=0 即 per-row）。"""

    axis: int


@dataclass(frozen=True)
class PerRow(Granularity):
    """dim 是行维，其余维归约；2D 权重上等价于 PerAxis(0)。"""

    dim: int = -1


def get_block_size(input_shape: Tuple[int, ...], granularity: Granularity) -> Tuple[int, ...]:
    """移植自 torchao/quantization/utils.py::get_block_size。

    输入：输入张量形状、粒度对象。
    输出：与 input_shape 等长的 block_size，元素 1 表示该维保留、等于该维长度表示归约。
    预期行为：PerTensor 归约全部维；PerAxis 保留 axis 维；PerRow 归约 dim 维。
    """
    if isinstance(granularity, PerTensor):
        return input_shape
    elif isinstance(granularity, PerAxis):
        block_size = list(input_shape)
        block_size[granularity.axis] = 1
        return tuple(block_size)
    elif isinstance(granularity, PerRow):
        block_size = [1] * len(input_shape)
        block_size[granularity.dim] = input_shape[granularity.dim]
        return tuple(block_size)
    raise ValueError(f"Unsupported Granularity: {granularity}")


# ---------------------------------------------------------------- 原语
# 以下全部移植自 torchao/quantization/quant_primitives.py


class MappingType(enum.Enum):
    """浮点到整数的映射方式（只留本模型用的 SYMMETRIC）。"""

    SYMMETRIC = "symmetric"


class ZeroPointDomain(enum.Enum):
    """zero-point 所在域（只留本模型用的 NONE）。"""

    INT = "int"
    NONE = "none"


class _Round(torch.autograd.Function):
    """torchao 原样移植：取整的 STE（前向 round、反向直通）。

    输入：任意 float 张量。
    输出：同形状的取整结果。
    预期行为：反向把梯度原样传回；前向是普通 torch.round。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, gy: torch.Tensor) -> torch.Tensor:
        return gy


class _ClampSTE(torch.autograd.Function):
    """torchao 原样移植：保住边界梯度的 clamp。

    输入：待夹取张量、上下界。
    输出：夹取后的张量。
    预期行为：反向用 torch.where 选择，落在界内的位置原样传梯度、界外给 0；
              不用乘法掩码，免得 out-of-range 处的 NaN 梯度被 0 * nan 污染。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.lo, ctx.hi = lo, hi
        return torch.clamp(x, lo, hi)

    @staticmethod
    def backward(ctx, gy: torch.Tensor):
        (x,) = ctx.saved_tensors
        in_range = (x >= ctx.lo) & (x <= ctx.hi)
        zero = torch.zeros((), dtype=gy.dtype, device=gy.device)
        return torch.where(in_range, gy, zero), None, None


def _get_and_check_qmin_qmax(dtype, quant_min, quant_max):
    """torchao 原样移植：按 dtype 取默认量化范围并校验上界。

    输入：量化 dtype、可选的 quant_min / quant_max。
    输出：校验过的 (quant_min, quant_max)。
    预期行为：dtype 不在表里就抛 ValueError；越界就 assert 失败。
    """
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise ValueError(f"Unsupported dtype: {dtype}")
    quant_min_lower_bound, quant_max_upper_bound = _DTYPE_TO_QVALUE_BOUNDS[dtype]
    if quant_min is None:
        quant_min = quant_min_lower_bound
    if quant_max is None:
        quant_max = quant_max_upper_bound

    assert quant_min >= quant_min_lower_bound, (
        "quant_min out of bound for dtype, "
        f"quant_min_lower_bound: {quant_min_lower_bound} quant_min: {quant_min}"
    )
    assert quant_max <= quant_max_upper_bound, (
        "quant_max out of bound for dtype, "
        f"quant_max_upper_bound: {quant_max_upper_bound} quant_max: {quant_max}"
    )
    return quant_min, quant_max


def _get_reduction_params(block_size, input_size):
    """torchao 原样移植：由 block_size 推出 view 形状与归约维。

    输入：block_size、input_size（等长）。
    输出：(shape_for_reduction, reduction_dims)。
    预期行为：block_size[i] == 1 的维保留；block_size[i] == input_size[i] 的维归约；
              两者都不满足时要求整除，并拆成 (input//block, block) 两维后归约后一维。
    """
    assert len(block_size) == len(input_size)
    shape_for_reduction = []
    reduction_dims = []
    cur_dim = 0
    for i in range(len(block_size)):
        if block_size[i] != input_size[i] and block_size[i] > 1:
            assert input_size[i] % block_size[i] == 0, (
                f"Expecting input size at {i} dimension: {input_size[i]} to be divisible by block_size at {i} dimension: {block_size[i]}"
            )
            shape_for_reduction.append(input_size[i] // block_size[i])
            shape_for_reduction.append(block_size[i])
            # reduce over the block_size[i] dim
            reduction_dims.append(cur_dim + 1)
            cur_dim += 2
        else:
            # block_size[i] == input_size[i] or block_size[i] == 1
            shape_for_reduction.append(input_size[i])
            # we only need to reduce over the dimension if block_size is greater than 1
            # otherwise it's already the same as reduced dimension
            if block_size[i] != 1:
                reduction_dims.append(cur_dim)
            cur_dim += 1
    return shape_for_reduction, reduction_dims


def _quantize_affine_no_zero_point_no_dtype_cast(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
) -> torch.Tensor:
    """torchao 原样移植：无 zero-point 的仿射量化（不做 dtype 转换）。

    输入：float 张量、block_size、scale、zero_point（本口径下为 None）。
    输出：取整并夹取后的 float 张量，值域是 scale 的整数倍。
    预期行为：q = clamp(round(x / scale), quant_min, quant_max)；取整走 _Round 的 STE、
              夹取走 _ClampSTE，两者都不会在 torch.compile 图里被折叠掉。
    """
    assert input.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Unsupported input dtype: {input.dtype}"
    assert len(block_size) == input.dim(), (
        f"Got input dim:{input.dim()}, block_size: {block_size}"
    )
    shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input.size()
    )
    original_shape = input.shape
    input = input.view(shape_for_reduction)
    shape_after_reduction = shape_for_reduction
    for i in reduction_dims:
        shape_after_reduction[i] = 1
    scale = scale.view(shape_after_reduction)

    if zero_point is not None and zero_point.numel() > 0:
        zero_point = zero_point.view(shape_after_reduction)
    else:
        # in some cases zero_point being a non-value shows as a tensor
        # with numel=0 which we handle by unifying the two
        zero_point = None

    quant = _ClampSTE.apply(_Round.apply(input * (1.0 / scale)), quant_min, quant_max)
    quant = quant.view(original_shape)

    return quant


def _dequantize_affine_no_zero_point_no_dtype_check(
    input: torch.Tensor,
    block_size: List[int],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_min: Union[int, float],
    quant_max: Union[int, float],
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """torchao 原样移植：无 zero-point 的反量化（不校验 dtype）。

    输入：量化后的张量、block_size、scale、zero_point（必须为 None）。
    输出：反量化回 output_dtype 的张量。
    预期行为：dq = q * scale（无 zero-point 相减）；形状按 block_size 广播回原形状。
    """
    assert len(block_size) == input.dim(), (
        f"Got input dim:{input.dim()}, block_size: {block_size}"
    )
    shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input.size()
    )
    original_shape = input.shape
    input = input.view(shape_for_reduction)
    shape_after_reduction = shape_for_reduction
    for i in reduction_dims:
        shape_after_reduction[i] = 1
    scale = scale.view(shape_after_reduction)

    assert zero_point is None, (
        "zero_point should be None for _dequantize_affine_no_zero_point"
    )
    dequant = input.to(output_dtype)
    dequant = dequant * scale

    return dequant.view(original_shape).to(output_dtype)


@torch.no_grad()
def _choose_qparams_affine(
    input: torch.Tensor,
    mapping_type: str,
    block_size: List[int],
    target_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    eps: Optional[float] = None,
    scale_dtype: Optional[torch.dtype] = None,
    zero_point_dtype: Optional[torch.dtype] = None,
    keepdim: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """torchao 移植（只留 SYMMETRIC 分支）：算对称量化的 scale / zero_point。

    输入：float 张量、映射方式名、block_size、目标 dtype、可选范围与 eps。
    输出：(scale, zero_point)；对称口径下 zero_point 恒为 0。
    预期行为：scale = max(|x|) / ((quant_max - quant_min) / 2)，夹到 eps 以上；
              quant_min=-127、quant_max=127 时正好是 PLAN §7.1 的 max/127。
    """
    quant_min, quant_max = _get_and_check_qmin_qmax(target_dtype, quant_min, quant_max)
    assert mapping_type in [MappingType.SYMMETRIC.name], (
        f"Unsupported mapping type: {mapping_type}"
    )

    if scale_dtype is None:
        scale_dtype = input.dtype
    if eps is None:
        eps = torch.finfo(input.dtype).smallest_normal

    assert len(block_size) == input.dim(), (
        f"Got input dim:{input.dim()}, block_size: {block_size}"
    )
    original_input_size = input.size()
    shape_for_reduction, reduction_dims = _get_reduction_params(
        block_size, input.size()
    )
    input = input.view(shape_for_reduction)

    min_val = torch.amin(input, dim=reduction_dims, keepdim=keepdim)
    max_val = torch.amax(input, dim=reduction_dims, keepdim=keepdim)

    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

    # scales
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scale = max_val_pos / (float(quant_max - quant_min) / 2)
    zero_point = torch.full_like(scale, int((quant_max + quant_min + 1) / 2))
    scale = torch.clamp(scale, min=eps)

    if keepdim:
        output_shape = [
            original_input_size[i] // block_size[i] for i in range(len(block_size))
        ]
        scale = scale.reshape(output_shape)
        zero_point = zero_point.reshape(output_shape)

    return scale.to(dtype=scale_dtype, device=input.device), zero_point.to(
        dtype=zero_point_dtype
    )


@torch.no_grad()
def choose_qparams_affine(
    input: torch.Tensor,
    mapping_type: MappingType,
    block_size: Tuple[int],
    target_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    eps: Optional[float] = None,
    scale_dtype: Optional[torch.dtype] = None,
    zero_point_dtype: Optional[torch.dtype] = torch.int32,
    keepdim: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """torchao 原样移植：choose_qparams_affine 的公开签名。

    输入：同 _choose_qparams_affine，mapping_type 用枚举。
    输出：(scale, zero_point)。
    预期行为：直接转调 _choose_qparams_affine，把枚举换成名字。
    """
    return _choose_qparams_affine(
        input,
        mapping_type.name,
        block_size,
        target_dtype,
        quant_min,
        quant_max,
        eps,
        scale_dtype,
        zero_point_dtype,
        keepdim,
    )


def _do_fake_quantize_affine(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """torchao 移植（只留 INT / NONE 两个域）：量化 + 反量化。

    输入：float 张量、block_size、scale、zero_point、目标 dtype、范围、zero-point 域。
    输出：(取整后的整数张量, 反量化回输入 dtype 的张量)。
    预期行为：NONE 域走 no_zero_point 两个算子，并要求 zero_point 为 None。
    """
    input_dtype = input.dtype
    quant_min, quant_max = _get_and_check_qmin_qmax(quant_dtype, quant_min, quant_max)
    if zero_point_domain == ZeroPointDomain.INT:
        _quantize_affine = _quantize_affine_no_dtype_cast
        _dequantize_affine = _dequantize_affine_no_dtype_check
    elif zero_point_domain == ZeroPointDomain.NONE:
        _quantize_affine = _quantize_affine_no_zero_point_no_dtype_cast
        _dequantize_affine = _dequantize_affine_no_zero_point_no_dtype_check
    else:
        raise ValueError(f"Unrecognized zero point domain: {zero_point_domain}")
    q = _quantize_affine(
        input,
        block_size,
        scale,
        zero_point,
        quant_min,
        quant_max,
    )
    dq = _dequantize_affine(
        q,
        block_size,
        scale,
        zero_point,
        quant_min,
        quant_max,
        output_dtype=input_dtype,
    )
    return (q, dq)


def _fake_quantize_affine(
    input: torch.Tensor,
    block_size: Tuple[int, ...],
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    quant_dtype: torch.dtype,
    quant_min: Optional[Union[int, float]] = None,
    quant_max: Optional[Union[int, float]] = None,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
) -> torch.Tensor:
    """torchao 原样移植：QAT 用的假量化算子。

    输入：同 _do_fake_quantize_affine。
    输出：反量化回输入 dtype 的张量（前向等价于 quantize + dequantize，不做 dtype 转换）。
    预期行为：反向由 _Round / _ClampSTE 的 STE 给出。
    """
    (_, fq) = _do_fake_quantize_affine(
        input,
        block_size,
        scale,
        zero_point,
        quant_dtype,
        quant_min,
        quant_max,
        zero_point_domain,
    )
    return fq


def _fake_quantize_per_channel_group(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor],
    quant_min: int,
    quant_max: int,
    group_size: int,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
) -> torch.Tensor:
    """移植自 torchao/quantization/qat/utils.py::_fake_quantize_per_channel_group。

    输入：2D 输入、scale、zero_point、范围、组宽。
    输出：假量化后的张量。
    预期行为：block_size 固定为 (1, group_size)，即最后一维分组、每组一组 qparam。
    """
    assert group_size > 1
    assert input.shape[-1] % group_size == 0
    assert input.dim() == 2
    block_size = (1, group_size)
    return _fake_quantize_affine(
        input,
        block_size,
        scales,
        zero_points,
        quant_dtype=torch.int32,
        quant_min=quant_min,
        quant_max=quant_max,
        zero_point_domain=zero_point_domain,
    )


# ---------------------------------------------------------------- 配置与量化器
# 移植自 torchao/quantization/qat/fake_quantize_config.py 与 fake_quantizer.py


@dataclass
class IntxFakeQuantizeConfig:
    """移植自 torchao.quantization.qat.fake_quantize_config.IntxFakeQuantizeConfig。

    输入：dtype / granularity / mapping_type / scale 与 zero_point 精度 / zero-point 域 /
          是否动态 / eps / quant_min / quant_max。
    输出：可直接构造 IntxFakeQuantizer 的配置对象。
    预期行为：quant_min / quant_max 未给时按 dtype 取默认上界（int8 是 -128..127）；
              本模型统一显式传 -127 / 127（PLAN §7.1）。
    """

    dtype: torch.dtype
    granularity: Granularity
    mapping_type: MappingType
    scale_precision: torch.dtype
    zero_point_precision: torch.dtype
    zero_point_domain: ZeroPointDomain
    is_dynamic: bool = True
    eps: Optional[float] = None
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None

    def __post_init__(self):
        """输入：无；输出：无。预期行为：把缺省的 quant_min/quant_max 补成 dtype 默认值。"""
        self.quant_min, self.quant_max = _get_and_check_qmin_qmax(
            self.dtype, self.quant_min, self.quant_max
        )


def per_tensor_int8() -> IntxFakeQuantizeConfig:
    """输入：无；输出：激活/逐元素常量的口径——int8、per-tensor、对称、动态。"""
    return IntxFakeQuantizeConfig(
        dtype=torch.int8,
        granularity=PerTensor(),
        mapping_type=MappingType.SYMMETRIC,
        scale_precision=torch.float32,
        zero_point_precision=torch.int32,
        zero_point_domain=ZeroPointDomain.NONE,
        is_dynamic=True,
        quant_min=INT8_MIN,
        quant_max=INT8_MAX,
    )


def per_row_int8() -> IntxFakeQuantizeConfig:
    """输入：无；输出：权重的口径——int8、per-row（per-channel）、对称、动态。"""
    return IntxFakeQuantizeConfig(
        dtype=torch.int8,
        granularity=PerRow(-1),
        mapping_type=MappingType.SYMMETRIC,
        scale_precision=torch.float32,
        zero_point_precision=torch.int32,
        zero_point_domain=ZeroPointDomain.NONE,
        is_dynamic=True,
        quant_min=INT8_MIN,
        quant_max=INT8_MAX,
    )


class FakeQuantizerBase(torch.nn.Module):
    """移植自 torchao.quantization.qat.fake_quantizer.FakeQuantizerBase。"""

    config: IntxFakeQuantizeConfig

    def __repr__(self) -> str:
        """输入：无；输出：带配置的可读字符串。"""
        return "FakeQuantizer(%s)" % self.config

    @staticmethod
    def from_config(config: IntxFakeQuantizeConfig) -> "FakeQuantizerBase":
        """输入：配置；输出：对应的量化器模块。"""
        return IntxFakeQuantizer(config)


class IntxFakeQuantizer(FakeQuantizerBase):
    """移植自 torchao.quantization.qat.fake_quantizer.IntxFakeQuantizer。

    输入：forward(x) 吃任意 float 张量。
    输出：假量化后的张量（同形状、同 dtype）。
    预期行为：`enabled=False` 时原样返回，保证非 QAT 路径与不开量化的模型逐位一致；
              `enabled=True` 时按 granularity 走 per-tensor 或 per-row 假量化。
    """

    def __init__(self, config: IntxFakeQuantizeConfig, enabled: bool = True):
        super().__init__()
        self.config = config
        self.enabled = enabled
        self.scale: Optional[torch.Tensor] = None
        self.zero_point: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入：float 张量；输出：假量化后的张量。"""
        if not self.enabled:
            return x
        if isinstance(self.config.granularity, PerTensor):
            return self._per_tensor_forward(x)
        elif isinstance(self.config.granularity, (PerAxis, PerRow)):
            return self._per_channel_or_group_forward(x)
        raise ValueError("Unknown granularity '%s'" % self.config.granularity)

    def _per_tensor_forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入：float 张量；输出：per-tensor 假量化结果。"""
        qmin, qmax = self.config.quant_min, self.config.quant_max
        block_size = get_block_size(x.shape, self.config.granularity)
        if self._should_compute_qparams():
            scale, zero_point = choose_qparams_affine(
                x,
                mapping_type=self.config.mapping_type,
                block_size=block_size,
                target_dtype=self.config.dtype,
                quant_min=qmin,
                quant_max=qmax,
                eps=self.config.eps,
                scale_dtype=self.config.scale_precision,
                zero_point_dtype=self.config.zero_point_precision,
            )
            self.scale = scale
            self.zero_point = (
                None if self.config.zero_point_domain is ZeroPointDomain.NONE else zero_point
            )
        return _fake_quantize_affine(
            x,
            block_size,
            self.scale,
            self.zero_point,
            self.config.dtype,
            qmin,
            qmax,
            self.config.zero_point_domain,
        )

    def _per_channel_or_group_forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入：2D float 张量；输出：per-row 假量化结果。"""
        granularity = self.config.granularity
        if isinstance(granularity, PerAxis):
            assert granularity.axis == 0
            group_size = x.size()[-1]
        elif isinstance(granularity, PerRow):
            assert granularity.dim == -1
            group_size = x.size()[-1]
        else:
            raise ValueError("Unexpected granularity '%s'" % granularity)

        if self._should_compute_qparams():
            scale, zero_point = choose_qparams_affine(
                x,
                mapping_type=self.config.mapping_type,
                block_size=(1, group_size),
                target_dtype=self.config.dtype,
                quant_min=self.config.quant_min,
                quant_max=self.config.quant_max,
                eps=self.config.eps,
                scale_dtype=self.config.scale_precision,
                zero_point_dtype=self.config.zero_point_precision,
            )
            self.scale = scale
            self.zero_point = (
                None if self.config.zero_point_domain is ZeroPointDomain.NONE else zero_point
            )
            if self.zero_point is not None:
                self.zero_point = self.zero_point.to(self.config.zero_point_precision)

        qmin, qmax = self.config.quant_min, self.config.quant_max
        return _fake_quantize_per_channel_group(
            x,
            self.scale,
            self.zero_point,
            qmin,
            qmax,
            group_size,
            self.config.zero_point_domain,
        )

    def _should_compute_qparams(self) -> bool:
        """输入：无；输出：是否要重算 qparam。

        预期行为：动态量化每次前向都重算（PLAN §7.1 的「运行时求 max」）；静态量化只在
                  首次前向算一次。
        """
        return (
            self.config.is_dynamic
            or self.scale is None
            or self.zero_point is None
            or self.scale.numel() == 0
            or (self.zero_point is not None and self.zero_point.numel() == 0)
        )


# ---------------------------------------------------------------- 模块级替换
# 移植自 torchao/quantization/qat/linear.py 与 embedding.py


class FakeQuantizedLinear(torch.nn.Linear):
    """移植自 torchao.quantization.qat.linear.FakeQuantizedLinear。

    输入：forward(x)，x 形状 (..., in_features)。
    输出：(..., out_features)。
    预期行为：激活先过 activation_fake_quantizer，权重过 weight_fake_quantizer，再 F.linear；
              bias 保持原样（本模型恒为 None）。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_config: Optional[IntxFakeQuantizeConfig] = None,
        weight_config: Optional[IntxFakeQuantizeConfig] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(in_features, out_features, bias, *args, **kwargs)
        self.activation_fake_quantizer = (
            FakeQuantizerBase.from_config(activation_config)
            if activation_config is not None
            else None
        )
        self.weight_fake_quantizer = (
            FakeQuantizerBase.from_config(weight_config)
            if weight_config is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入：x；输出：假量化权重/激活后的线性结果。"""
        if self.activation_fake_quantizer is not None:
            x = self.activation_fake_quantizer(x)
        if self.weight_fake_quantizer is not None:
            w = self.weight_fake_quantizer(self.weight)
        else:
            w = self.weight
        return F.linear(x, w, self.bias)

    @classmethod
    def from_linear(
        cls,
        mod: torch.nn.Linear,
        activation_config: Optional[IntxFakeQuantizeConfig] = None,
        weight_config: Optional[IntxFakeQuantizeConfig] = None,
    ) -> "FakeQuantizedLinear":
        """输入：原 nn.Linear 与两份配置；输出：共享同一 weight / bias 的假量化版。"""
        new_linear = FakeQuantizedLinear(
            mod.in_features,
            mod.out_features,
            mod.bias is not None,
            activation_config=activation_config,
            weight_config=weight_config,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
        )
        if mod.weight.device != torch.device("meta"):
            new_linear.weight = mod.weight
            new_linear.bias = mod.bias
        return new_linear


class FakeQuantizedEmbedding(torch.nn.Embedding):
    """移植自 torchao.quantization.qat.embedding.FakeQuantizedEmbedding。

    输入：forward(x)，x 是整数 id。
    输出：(..., embedding_dim)。
    预期行为：权重过 weight_fake_quantizer 再查表；输入是 one-hot 查表，不做激活量化。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        weight_config: Optional[IntxFakeQuantizeConfig] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            *args,
            **kwargs,
        )
        self.weight_fake_quantizer = (
            FakeQuantizerBase.from_config(weight_config)
            if weight_config is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入：整数 id 张量；输出：假量化权重查表结果。"""
        if self.weight_fake_quantizer is not None:
            w = self.weight_fake_quantizer(self.weight)
        else:
            w = self.weight
        return F.embedding(
            x,
            w,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    @classmethod
    def from_embedding(
        cls,
        mod: torch.nn.Embedding,
        weight_config: Optional[IntxFakeQuantizeConfig] = None,
    ) -> "FakeQuantizedEmbedding":
        """输入：原 nn.Embedding 与权重配置；输出：共享同一 weight 的假量化版。"""
        new_embedding = FakeQuantizedEmbedding(
            mod.num_embeddings,
            mod.embedding_dim,
            mod.padding_idx,
            mod.max_norm,
            mod.norm_type,
            mod.scale_grad_by_freq,
            mod.sparse,
            weight_config=weight_config,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
        )
        if mod.weight.device != torch.device("meta"):
            new_embedding.weight = mod.weight
        return new_embedding


def enable_linear_fake_quant(mod: torch.nn.Module) -> None:
    """移植自 torchao.quantization.qat.linear.enable_linear_fake_quant。

    输入：模块；输出：无。预期行为：打开 FakeQuantizedLinear 的两个量化器。
    """
    if isinstance(mod, FakeQuantizedLinear):
        if mod.activation_fake_quantizer is not None:
            mod.activation_fake_quantizer.enabled = True
        if mod.weight_fake_quantizer is not None:
            mod.weight_fake_quantizer.enabled = True


def enable_embedding_fake_quant(mod: torch.nn.Module) -> None:
    """移植自 torchao.quantization.qat.embedding.enable_embedding_fake_quant。

    输入：模块；输出：无。预期行为：打开 FakeQuantizedEmbedding 的权重量化器。
    """
    if isinstance(mod, FakeQuantizedEmbedding) and mod.weight_fake_quantizer is not None:
        mod.weight_fake_quantizer.enabled = True


def disable_linear_fake_quant(mod: torch.nn.Module) -> None:
    """输入：模块；输出：无。预期行为：关掉 FakeQuantizedLinear 的量化器。"""
    if isinstance(mod, FakeQuantizedLinear):
        if mod.activation_fake_quantizer is not None:
            mod.activation_fake_quantizer.enabled = False
        if mod.weight_fake_quantizer is not None:
            mod.weight_fake_quantizer.enabled = False


def disable_embedding_fake_quant(mod: torch.nn.Module) -> None:
    """输入：模块；输出：无。预期行为：关掉 FakeQuantizedEmbedding 的量化器。"""
    if isinstance(mod, FakeQuantizedEmbedding) and mod.weight_fake_quantizer is not None:
        mod.weight_fake_quantizer.enabled = False


def prepare_qat(model: torch.nn.Module) -> torch.nn.Module:
    """输入：NanoRWKV；输出：同一个模型对象（已原地替换成 QAT 版）。

    预期行为：所有 nn.Linear -> FakeQuantizedLinear（权重 per-row、激活 per-tensor）、
              nn.Embedding -> FakeQuantizedEmbedding（权重 per-row）、模型自带的
              IntxFakeQuantizer（逐元素参数与中间激活的插桩点）全部 enabled=True。
              权重张量原地搬过去，state_dict 的键名与初始化结果都不变。
    """
    act_cfg, w_cfg = per_tensor_int8(), per_row_int8()
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if type(child) is torch.nn.Linear:
                setattr(module, child_name,
                        FakeQuantizedLinear.from_linear(child, act_cfg, w_cfg))
            elif type(child) is torch.nn.Embedding:
                setattr(module, child_name,
                        FakeQuantizedEmbedding.from_embedding(child, w_cfg))
    for module in model.modules():
        enable_linear_fake_quant(module)
        enable_embedding_fake_quant(module)
        if isinstance(module, IntxFakeQuantizer):
            module.enabled = True
    return model


def disable_qat(model: torch.nn.Module) -> torch.nn.Module:
    """输入：模型；输出：同一个模型对象。

    预期行为：把所有 IntxFakeQuantizer 置回 enabled=False（含 FakeQuantizedLinear /
              FakeQuantizedEmbedding 内部的），用于对照实验；模块类型不换回 nn.Linear。
    """
    for module in model.modules():
        disable_linear_fake_quant(module)
        disable_embedding_fake_quant(module)
        if isinstance(module, IntxFakeQuantizer):
            module.enabled = False
    return model
