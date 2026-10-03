"""WKV-7 的 CUDA 路径：JIT 编译 wkv7_op.cpp + wkv7_cuda.cu，包成 autograd.Function。

输入/输出口径与 src/wkv7.py::wkv7_chunked 一致：
    logw (B,T,H,C) fp32 —— 外部已按 Q15 网格量化过的 log 衰减；
    q/k/v/a/b (B,T,H,C) bf16；chunk_len 整除 T；state_step <= 0 表示不做 state 网格量化。
输出：y (B,T,H,C) bf16。
预期行为：前后向都由 CUDA kernel 完成，state 的 int32 网格量化在 kernel 内按 chunk 边界执行。
"""

import os
import subprocess

import torch
from torch.utils.cpp_extension import load

_VCVARS = [r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
           r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"]


def _ensure_msvc_env():
    """输入：无；输出：无（把 MSVC 的编译环境变量注入本进程）。

    预期行为：Windows 上 torch.utils.cpp_extension 每次都自己去找 MSVC，在普通 shell 里
    会抛 DistutilsPlatformError。这里先跑一次 vcvars64.bat、把它 set 出来的变量收进
    os.environ，再打上 DISTUTILS_USE_SDK/MSSdk，setuptools 就直接用这套变量、不再去搜。
    非 Windows 或已经设过 DISTUTILS_USE_SDK 时直接返回。
    """
    if os.name != "nt" or os.environ.get("DISTUTILS_USE_SDK") == "1":
        return
    vcvars = next((path for path in _VCVARS if os.path.exists(path)), None)
    if vcvars is None:
        return
    out = subprocess.run('call "{0}" >nul && set'.format(vcvars), shell=True,
                         capture_output=True, text=True, errors="ignore").stdout
    for line in out.splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            os.environ[key] = value
    os.environ["DISTUTILS_USE_SDK"] = "1"
    os.environ["MSSdk"] = "1"

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD = os.path.join(os.path.dirname(_HERE), "out", "ext_cache")

_ext = None


def _load_ext():
    """输入：无；输出：编译好的扩展模块。首次调用就地编译，之后命中 build 目录缓存。"""
    global _ext
    if _ext is None:
        _ensure_msvc_env()
        os.makedirs(_BUILD, exist_ok=True)
        _ext = load(
            name="nanomeow_wkv7",
            sources=[os.path.join(_HERE, "wkv7_op.cpp"),
                     os.path.join(_HERE, "wkv7_cuda.cu")],
            extra_cuda_cflags=["-O3"],
            build_directory=_BUILD,
            verbose=False,
        )
    return _ext


class _WKV7Cuda(torch.autograd.Function):
    """WKV-7 递推的 CUDA 前后向；反向由 kernel 直接算，不经过 autograd 图。"""

    @staticmethod
    def forward(ctx, logw, q, k, v, a, b, chunk_len, state_step):
        _load_ext()
        bsz, t_len, n_head, dim = q.shape
        y = torch.empty_like(q)
        sa = torch.empty((bsz, t_len, n_head, dim), device=q.device, dtype=torch.float32)
        s = torch.empty((bsz, n_head, t_len // chunk_len, dim, dim),
                        device=q.device, dtype=torch.float32)
        torch.ops.nanomeow_wkv7.forward(logw, q, k, v, a, b, y, s, sa, chunk_len, state_step)
        ctx.save_for_backward(logw, q, k, v, a, b, s, sa)
        ctx.chunk_len = chunk_len
        state = s[:, :, -1]
        ctx.mark_non_differentiable(state)
        return y, state

    @staticmethod
    def backward(ctx, dy, dstate):
        # state 已 mark_non_differentiable：引擎回传的是全零占位，不是真实梯度
        logw, q, k, v, a, b, s, sa = ctx.saved_tensors
        dlogw = torch.empty_like(logw)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        torch.ops.nanomeow_wkv7.backward(logw, q, k, v, a, b, dy.contiguous(), s, sa,
                                         dlogw, dq, dk, dv, da, db, ctx.chunk_len)
        return dlogw, dq, dk, dv, da, db, None, None


def wkv7_cuda(logw, q, k, v, a, b, chunk_len=16, state_step=0.0, return_state=False):
    """输入：见模块 docstring；输出：y (B,T,H,C)，return_state=True 时额外给 state (B,H,C,C)。

    预期行为：T 必须整除 chunk_len；state_step <= 0 时不做 state 网格量化。
    state 是最后一个 chunk 边界上的递推状态（已按 state_step 取整），与 wkv7_chunked 同口径；
    它 detach 后返回（kernel 反向不含经 state 回流的那一项，模型层本来就只拿它做增量解码）。
    """
    assert q.shape[1] % chunk_len == 0, "T 必须整除 chunk_len"
    # state_step 以 0 维张量直接进 kernel：图里对张量调 float() 会触发 .item() 断图，
    # 而且 host 同步会拖慢每一步。传 Python float 时（测试/不开 QAT）才现造一个常量张量。
    if not torch.is_tensor(state_step):
        state_step = torch.full((), float(state_step), device=q.device, dtype=torch.float32)
    elif state_step.dim() != 0:
        state_step = state_step.reshape(())
    y, state = _WKV7Cuda.apply(logw.contiguous(), q.contiguous(), k.contiguous(),
                               v.contiguous(), a.contiguous(), b.contiguous(),
                               int(chunk_len), state_step.contiguous())
    return (y, state) if return_state else y