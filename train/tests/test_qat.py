"""QAT 测试：插桩点覆盖、关闭时与移植前逐位一致、STE 梯度、torch.compile 不折叠量化点。"""

import torch

from src.config import NanoConfig
from src.model import DEAD_BY_REFERENCE_BRANCH, NanoRWKV
from src.qat import (FakeQuantizedEmbedding, FakeQuantizedLinear, IntxFakeQuantizer,
                     disable_qat, per_row_int8, per_tensor_int8, prepare_qat)

TINY = NanoConfig()

# 模型自带的插桩点个数：每层 Tmix 4 个（act/param/w/wkv）+ CMix 2 个 + Block 2 个，
# 两层共 16 个，再加顶层的 fq_act / fq_param。
N_HOOKS = 18


def _batch(batch=2, ctx=16, vocab=256, seed=0):
    """输入：形状与种子；输出：(idx (B,T) int64, y (B,T) int64) 的随机样本。"""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, vocab, (batch, ctx + 1), generator=g)
    return idx[:, :-1].contiguous(), idx[:, 1:].contiguous()


def _loss(logits, y):
    """输入：logits (B,T,V)、y (B,T)；输出：标量交叉熵。"""
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def test_hooks_exist_and_start_disabled():
    """插桩点个数固定为 N_HOOKS，且默认全是关闭的（保证非 QAT 路径等于移植前）。"""
    model = NanoRWKV(TINY)
    hooks = [m for m in model.modules() if isinstance(m, IntxFakeQuantizer)]
    assert len(hooks) == N_HOOKS, f"插桩点数量变了：{len(hooks)} != {N_HOOKS}"
    assert not any(m.enabled for m in hooks)


def test_prepare_qat_covers_every_linear_embedding_and_hook():
    """PLAN §7.4：所有 nn.Linear / nn.Embedding 都换成 QAT 版，且每个插桩点都打开。"""
    model = prepare_qat(NanoRWKV(TINY))
    assert not any(type(m) is torch.nn.Linear for m in model.modules())
    assert not any(type(m) is torch.nn.Embedding for m in model.modules())
    assert any(isinstance(m, FakeQuantizedLinear) for m in model.modules())
    assert any(isinstance(m, FakeQuantizedEmbedding) for m in model.modules())
    hooks = [m for m in model.modules() if isinstance(m, IntxFakeQuantizer)]
    assert len(hooks) > N_HOOKS, "nn.Linear / nn.Embedding 内部的量化器没被算进来"
    assert all(m.enabled for m in hooks)


def test_qat_disabled_is_bitwise_identical_to_plain():
    """关掉 QAT 时前向必须与移植前逐位一致——插桩点是恒等映射。"""
    torch.manual_seed(0)
    model = NanoRWKV(TINY)
    idx, _ = _batch()
    with torch.no_grad():
        base, _ = model(idx)

    prepare_qat(model)
    disable_qat(model)
    with torch.no_grad():
        off, _ = model(idx)
    assert torch.equal(base, off), "关闭 QAT 后输出变了，插桩点不是恒等映射"


def test_qat_enabled_changes_output_and_stays_finite():
    """打开 QAT 后输出必须变（插桩点接上了）且保持有限。"""
    torch.manual_seed(0)
    model = NanoRWKV(TINY)
    idx, _ = _batch()
    with torch.no_grad():
        base, _ = model(idx)

    prepare_qat(model)
    with torch.no_grad():
        on, _ = model(idx)
    assert torch.isfinite(on).all()
    assert (on - base).abs().max() > 1e-3, "打开 QAT 后输出没变，插桩点没接上"


def test_qat_gradients_reach_every_parameter():
    """PLAN §10.3：QAT 打开后每个参数依然进计算图，且没有梯度恒为零的死参数。"""
    torch.manual_seed(0)
    model = prepare_qat(NanoRWKV(TINY))
    model.train()
    idx, y = _batch()

    logits, _ = model(idx)
    _loss(logits, y).backward()
    missing = [n for n, p in model.named_parameters()
               if p.grad is None and n not in DEAD_BY_REFERENCE_BRANCH]
    assert missing == [], f"这些参数没进计算图：{missing}"

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # 20 步的理由同 test_model.py：零初始化要靠旁路梯度解开，QAT 下逃逸慢约一倍。
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        logits, _ = model(idx)
        _loss(logits, y).backward()
        opt.step()

    opt.zero_grad(set_to_none=True)
    logits, _ = model(idx)
    _loss(logits, y).backward()
    dead = [n for n, p in model.named_parameters()
            if n not in DEAD_BY_REFERENCE_BRANCH and p.grad is not None
            and float(p.grad.norm()) == 0.0]
    assert dead == [], f"这些参数在 QAT 下梯度恒为零：{dead}"


def test_fake_quant_ste_passes_gradient():
    """STE：量化点在界内时梯度必须原样传回（不是 0）。"""
    q = IntxFakeQuantizer(per_tensor_int8())
    x = torch.randn(16, requires_grad=True)
    q(x).sum().backward()
    assert torch.all(x.grad == 1.0)


def test_per_row_quant_matches_manual():
    """per-row 对称 int8 必须等于手算：round(w / (max|w_row|/127)) 再乘回 scale。"""
    q = IntxFakeQuantizer(per_row_int8())
    w = torch.tensor([[0.0, 1.0, -2.0], [3.0, 0.0, 0.0]])
    scale = torch.tensor([[2.0 / 127.0], [3.0 / 127.0]])
    expect = torch.round(w / scale).clamp(-127, 127) * scale
    assert torch.allclose(q(w), expect, atol=1e-6)


def test_per_tensor_range_is_symmetric_127():
    """PLAN §7.1：对称、无 zero-point，范围必须是 [-127, 127] 而不是 [-128, 127]。"""
    q = IntxFakeQuantizer(per_tensor_int8())
    assert q.config.quant_min == -127 and q.config.quant_max == 127
    x = torch.tensor([[-1.0, 0.5]])
    out = q(x)
    # scale = max|x| / 127 = 1/127；-1.0 正好落在 -127 上，0.5 取整到 64
    assert torch.allclose(q.scale, torch.tensor(1.0 / 127.0))
    assert torch.allclose(out, torch.tensor([[-1.0, 64.0 / 127.0]]), atol=1e-6)


def test_compile_preserves_fake_quant():
    """PLAN §7.4：torch.compile 之后量化点不能被折叠成 no-op。

    判据：编译后的输出要等于 eager 的 QAT 输出，且**不等于**关掉量化的输出——
    后者正是 round/clamp 被折叠掉时会出现的症状。
    """
    torch.manual_seed(0)
    model = prepare_qat(NanoRWKV(TINY)).eval()
    idx, _ = _batch(batch=1, ctx=16)
    with torch.no_grad():
        eager_q = model(idx)[0]

    net = torch.compile(model)
    with torch.no_grad():
        compiled = net(idx)[0]

    disable_qat(model)
    with torch.no_grad():
        eager_plain = model(idx)[0]

    assert torch.allclose(compiled, eager_q, atol=1e-5, rtol=1e-4), "编译后与 eager 的 QAT 输出不一致"
    assert not torch.allclose(compiled, eager_plain, atol=1e-3), "编译后等于关量化的结果，量化点被折叠了"
