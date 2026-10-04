"""QAT 测试：插桩点覆盖、关闭时与移植前逐位一致、STE 梯度、torch.compile 不折叠量化点。"""

import torch

from src.config import NanoConfig
from src.model import DEAD_BY_REFERENCE_BRANCH, NanoRWKV
from src.qat import (FakeQuantizedEmbedding, FakeQuantizedLinear, IntxFakeQuantizer,
                     PerPosition, disable_qat, per_position_int8, per_row_int8,
                     per_tensor_int8, prepare_qat)

TINY = NanoConfig()

# 模型自带的插桩点个数：每层 Tmix 4 个（act/param/w/wkv）+ CMix 2 个 + Block 2 个，
# 两层共 16 个，再加顶层的 fq_act / fq_param。
N_HOOKS = 8 * TINY.n_layer + 2


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
    """所有 nn.Linear / nn.Embedding 都换成 QAT 版，且每个插桩点都打开。"""
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
    """QAT 打开后每个参数依然进计算图，且没有梯度恒为零的死参数。"""
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


def test_qat_static_decay_gradients_reach_every_parameter():
    """静态 decay（dynamic_decay=False）下同样不能有死参数：x_w/w1/w2 已不存在，其余都要动。"""
    torch.manual_seed(0)
    model = prepare_qat(NanoRWKV(NanoConfig(dynamic_decay=False)))
    model.train()
    idx, y = _batch()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        logits, _ = model(idx)
        _loss(logits, y).backward()
        opt.step()
    dead = [n for n, p in model.named_parameters()
            if n not in DEAD_BY_REFERENCE_BRANCH and p.grad is not None
            and float(p.grad.norm()) == 0.0]
    assert dead == [], f"静态 decay 下有梯度恒为零的参数：{dead}"
    assert not [n for n in model.state_dict() if n.endswith((".x_w", ".w1", ".w2"))]


def test_fake_quant_ste_passes_gradient():
    """STE：量化点在界内时梯度必须原样传回（不是 0）。"""
    q = IntxFakeQuantizer(per_tensor_int8())
    x = torch.randn(16, requires_grad=True)
    q(x).sum().backward()
    # 端点精确可表示、没有元素被 clamp；反向是解析式，所以恒等于 1，不带 (1/scale)*scale 的 ulp 误差
    assert torch.equal(x.grad, torch.ones_like(x.grad))


def test_fake_quant_ste_survives_all_zero_tensor_in_bf16():
    """全零张量（scale 退化成 smallest_normal）下 STE 必须照样传梯度，哪怕 gy 远小于 7.8e-3。

    回归点：把 quantize / dequantize 拆成两步时，中间梯度 `gy * scale` 会掉进次正规区间
    被刷成 0（bf16 的下限是 9.2e-41，对应 gy < 7.8e-3），再乘回 1 / scale 仍是 0。
    decay 的低秩对是零初始化，第一步就产生全零张量，于是 x_w / w1 / w2 永远拿不到梯度。
    """
    q = IntxFakeQuantizer(per_tensor_int8())
    x = torch.zeros(4, 8, dtype=torch.bfloat16, requires_grad=True)
    (q(x) * 1e-8).sum().backward()
    assert torch.equal(q.scale, torch.tensor(torch.finfo(torch.bfloat16).smallest_normal)), (
        f"这组输入没有触发退化 scale：{q.scale}")
    assert torch.allclose(x.grad, torch.full_like(x.grad, 1e-8)), (
        f"全零张量下梯度被吃掉了：{x.grad.flatten()[:4]}")


def test_per_row_fake_quant_ste_survives_all_zero_weight():
    """per-row 权重全零（scale 同样退化成 smallest_normal）时也必须传梯度。"""
    q = IntxFakeQuantizer(per_row_int8())
    w = torch.zeros(32, 8, dtype=torch.bfloat16, requires_grad=True)
    (q(w.transpose(0, 1)) * 1e-8).sum().backward()
    assert (w.grad != 0).all(), "全零权重的梯度被吃掉了"


def test_per_row_quant_matches_manual():
    """per-row 对称 int8 必须等于手算：正负端各自除以自己的边界，取大的那个作 scale。"""
    q = IntxFakeQuantizer(per_row_int8())
    w = torch.tensor([[0.0, 1.0, -2.0], [3.0, 0.0, 0.0]])
    # 第 0 行负端 2/128 大于正端 1/127，取它；第 1 行只有正端 3/127
    scale = torch.tensor([[2.0 / 128.0], [3.0 / 127.0]])
    expect = torch.round(w / scale).clamp(-128, 127) * scale
    assert torch.allclose(q(w), expect, atol=1e-6)
    assert torch.allclose(q.scale.flatten(), scale.flatten(), atol=1e-6)


def test_per_tensor_range_matches_torchao_int8():
    """对称、无 zero-point，范围取 torchao 的 int8 默认 [-128, 127]，且端点必须精确可表示。"""
    q = IntxFakeQuantizer(per_tensor_int8())
    assert q.config.quant_min == -128 and q.config.quant_max == 127
    x = torch.tensor([[-1.0, 0.5]])
    out = q(x)
    # 负端 1/128 大于正端 0.5/127，取它；两端分别精确落在 -128 和 +64 上，没有饱和
    assert torch.allclose(q.scale, torch.tensor(1.0 / 128.0))
    assert torch.allclose(out, torch.tensor([[-1.0, 0.5]]), atol=1e-6)


def test_scale_arithmetic_is_fp32_not_input_dtype():
    """scale 的算术必须在 fp32 里做，不能落回 input.dtype。

    判据：取 max/127 不能精确落在 bf16 网格上的输入，算出的 scale 必须**不在** bf16 网格上。
    回归点：算术一旦落回 input.dtype，eager 会先把 scale 做 bf16 舍入，而 torch.compile 会把
    这一步提升到 fp32 —— 两条路得到不同的量化步长，整张量化网格挪一格（实测 bf16 输入下
    448 万个元素不同，scale 0.044921875 vs 0.04502952844）。
    """
    q = IntxFakeQuantizer(per_tensor_int8())
    x = torch.full((4, 8), 5.71875, dtype=torch.bfloat16)
    q(x)

    expect = torch.tensor(5.71875, dtype=torch.float32) / 127
    assert not torch.equal(expect, expect.to(torch.bfloat16).to(torch.float32)), (
        "这组输入的 scale 在 bf16 下可精确表示，测不出回归，换一组输入")
    assert q.scale.dtype == torch.float32
    assert torch.equal(q.scale, expect), (
        f"scale={q.scale.item()}，应为 fp32 的 {expect.item()}：算术被按 input.dtype 做了")


def test_compile_preserves_fake_quant():
    """torch.compile 之后量化点不能被折叠成 no-op。

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


def _spread_activations(B=2, T=5, C=8, seed=7):
    """输入：形状与种子；输出：每个位置量级差 6 倍的 (B,T,C) 激活。

    预期行为：位置之间量级拉开，才能看出「整批一个 scale」和「每位置一个 scale」的差别。
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, C, generator=g)
    return x * torch.linspace(0.05, 3.0, T).view(1, T, 1)


def test_activation_quant_matches_single_position_deployment():
    """激活插桩点必须与推理侧同口径：整批前向里第 t 个位置的量化结果，
    等于把该位置单独喂进去（T=1）时的结果。

    预期行为：部署一次只喂一个位置，per-tensor 在那里就是「一条向量一个 scale」；
              训练侧按位置分组才与它等价，否则 scale 被整个 batch 摊粗。
    """
    x = _spread_activations()
    q_batch = IntxFakeQuantizer(per_position_int8())(x)
    for b in range(x.shape[0]):
        for t in range(x.shape[1]):
            q_single = IntxFakeQuantizer(per_tensor_int8())(x[b, t])
            assert torch.equal(q_batch[b, t], q_single), (b, t)


def test_batch_wide_activation_quant_is_coarser_than_deployment():
    """对照：整批共用一个 scale（改之前的口径）会把小量级位置摊粗，与部署侧不一致。

    预期行为：这条是「为什么必须改」的证据，不是实现细节 —— 它一挂说明这个 bug 回来了。
    """
    x = _spread_activations()
    q_batch = IntxFakeQuantizer(per_tensor_int8())(x)
    q_pos = IntxFakeQuantizer(per_position_int8())(x)
    flat = int(x.abs().mean(dim=-1).argmin())
    tiny = (flat // x.shape[1], flat % x.shape[1])
    assert not torch.equal(q_batch[tiny], q_pos[tiny])
    assert torch.equal(q_pos[tiny], IntxFakeQuantizer(per_tensor_int8())(x[tiny]))


def test_prepare_qat_linear_activation_matches_deployment():
    """nn.Linear 的激活量化必须与 fq_act 同口径（逐位置）。

    预期行为：部署侧 GEMV 直接用输入自带的码与 scale、不再量化输入，
              所以训练侧这里也只能是逐位置；一挂说明口径又跑回 per-tensor 了。
    """
    model = NanoRWKV(TINY)
    prepare_qat(model)
    lin = model.blocks[0].att.receptance
    assert isinstance(lin.activation_fake_quantizer.config.granularity, PerPosition)


def test_requantizing_a_quantized_activation_is_idempotent_only_per_position():
    """输入：已经逐位置量化过的激活；输出：再量化一次的结果。

    预期行为：逐位置再量化必须逐位不变（部署侧不做这一步，训练侧做了也不能改值）；
              对照的 per-tensor 再量化必须改值，否则这条护栏没有鉴别力。
    """
    x = _spread_activations()
    q = IntxFakeQuantizer(per_position_int8())(x)
    assert torch.equal(IntxFakeQuantizer(per_position_int8())(q), q)
    assert not torch.equal(IntxFakeQuantizer(per_tensor_int8())(q), q)
