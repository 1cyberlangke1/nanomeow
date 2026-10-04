# train — nanomeow 模型训练与 INT8 量化导出流水线

用仓库自带的合成语料，训练一个可导出为 INT8 的 RWKV-7 x070 缩维模型：FP32 master 权重 + bf16 autocast
前向/反向 + `torch.compile` 加速 + QAT 全参数微调；导出脚本会把「导出的码 × scale」和训练侧的假量化
**逐位对拍**，对不上就直接报错退出。

---

## 模型与训练规格

| 维度 | 参数 / 规格 | 说明 |
| :--- | :--- | :--- |
| **基础架构** | RWKV-7 x070 | 3 层 / `n_embd=32` / `head_size=8` / `n_head=4` / `dim_ffn=32` / `dim_lora=8`，总参数量 42,912 |
| **分词与上下文** | Byte-level Tokenizer (V=256) | 无 BPE；`ctx_len=512` 字节 |
| **提示词模板** | `user:<输入>\nbot:<输出>` | 冒号后无空格；序列结束标记 `<ETX>` (0x03)，批次补齐用 `PAD_ID` (0x00) |
| **数值精度** | FP32 master + bf16 autocast | 权重以 FP32 保存，前向/反向走 bf16；WKV7 递推内部强制 FP32 |
| **量化感知训练** | 对称 INT8 假量化 | 权重 per-row（输出通道）、激活 per-tensor 动态；按 `pytorch/ao`（torchao）的做法 |
| **WKV7 算子** | 纯 PyTorch 分块递推 | `head_size=8` / `wkv_chunk=16` 且从零状态开始时启用 CUDA 快路径（JIT 编译，数值与分块版一致） |
| **编译加速** | `torch.compile` 默认开启 | 编译缓存落 `out/compile_cache`（`conftest.py` 里设的 `TORCHINDUCTOR_CACHE_DIR`） |

---

## 模块架构与源文件分布

| 模块路径 | 说明 |
| :--- | :--- |
| `src/config.py` | `NanoConfig` 模型结构定义（含 `param_count()` 参数量对账公式）与 `TrainConfig` 超参 dataclass |
| `src/model.py` | `NanoRWKV` 主体：从 `tmp/Mini_RWKV_7` 逐段移植 x070，再加上 QAT 插桩点；`RWKVState` 保存逐 token 推理时的跨步状态 |
| `src/qat.py` | 从 torchao 移植的量化 / QAT（只保留对称 INT8 分支）：量化粒度、Scale 求解、取整 STE、`FakeQuantizedLinear` / `FakeQuantizedEmbedding` |
| `src/wkv7/wkv7.py` | 分块 WKV7 递推的纯 PyTorch 实现 |
| `src/wkv7/wkv7_cuda.py` | CUDA 快路径的 JIT 编译包装（`wkv7_op.cpp` + `wkv7_cuda.cu` 是算子本体） |
| `src/tokenizer.py` | 字节级 tokenizer、`<ETX>` / `PAD_ID` 常量、流式 UTF-8 解码器 |
| `src/data.py` | 语料打包：补 `<ETX>`、右补 PAD、仅回答段打 Loss Mask |
| `src/train.py` | 训练入口（`--stage` 选阶段，当前语料走 `sft`） |
| `src/generate.py` | 主机侧生成：模板拼接 + 逐字节自回归 + 增量 UTF-8 解码（不吐 U+FFFD） |
| `scripts/export_int8.py` | Checkpoint → `infer/model_weights.h` + `infer/model_cfg.h`，自带逐位自检 |
| `notebooks/prepare_dataset.ipynb` | 语料清洗与统计体检 → `dataset/nana_clean.jsonl`（幂等、可断点续跑） |
| `data/chitchat_para.jsonl` | 仓库自带的合成寒暄语料：241 个规范答案、共 6908 条 |
| `tests/` | 62 条自动化用例（`test_data` 5 / `test_export_int8` 4 / `test_generate` 13 / `test_model` 13 / `test_qat` 17 / `test_wkv7` 10） |

---

## 数据处理

### 1. 语料清洗

在 Jupyter 或 VS Code 里从上到下把 `notebooks/prepare_dataset.ipynb`（3 格）跑一遍，产出
`dataset/nana_clean.jsonl`（单字段 `{"text": "user:<问>\nbot:<答>"}`）。每格都自带幂等检查：
跑过的步骤直接跳过，改了清洗参数就先删掉对应的 `*_clean.jsonl` 再重跑。

### 2. SFT 打包与 Loss Mask

`src/data.py` 在打包时做三件事：

1. 每条样本末尾补一个 `<ETX>` (0x03) 作为序列结束；
2. 整条样本（含 `<ETX>`）UTF-8 字节数超过 `ctx_len` 的直接丢掉，其余的右补 `PAD_ID`；
3. **Loss Mask 只覆盖回答段**（`\nbot:` 之后到 `<ETX>` 之前）——模板前缀由调用方在推理时给出，
   补齐位也不能参与 loss（否则模型会学着吐 NUL 字节）。

中间产物落在 `dataset/built/sft/`：`sft_x.npy` / `sft_y.npy` / `sft_mask.npy` / `sft_meta.json`（不入库）。

---

## 训练

### 1. 启动训练

```powershell
cd train
..\.venv\Scripts\python.exe -m src.train --stage sft

# 接着已有权重继续训练（换一轮超参或换 run 目录）
..\.venv\Scripts\python.exe -m src.train --stage sft --init-from out\sft_fix3\sft_step3000.pth --out out\sft_v2
```

关键默认值：`--batch-size 64`、`--ctx-len 512`、`--lr 6e-4`、`--min-lr 2e-5`、`--warmup-steps 10`、
`--weight-decay 0.001`、`--grad-clip 1.0`、`--qat 1`、`--compile 1`；学习率先 warmup，再按
cosine 衰减到 `--min-lr`。`--epochs` 决定跑几遍数据，`--steps` 是硬上限（0 表示交给 `--epochs` 按数据量算）。

### 2. 训练产物

`out/<run>/` 下：

| 文件 | 内容 |
| :--- | :--- |
| `<stage>.pth` | 最终权重（含 `step` 与模型配置） |
| `<stage>_stepN.pth` | 每 `--save-every` 步存一次的快照 |
| `<stage>_train.json` | stage / steps / tokens / seconds / final_loss / args / cfg 全量记录 |

---

## 量化导出

```powershell
.\.venv\Scripts\python.exe scripts\export_int8.py --ckpt out\sft_fix3\sft_step3000.pth
.\.venv\Scripts\python.exe ..\infer\c\tools\gen_weights.py
```

- 第一条：写 `infer/model_weights.h`（INT8 码 + 每行 int32 乘子 + int8 移位）和
  `infer/model_cfg.h`（模型结构常量，路径用 `--out` 改）；自检会把「导出的码 × scale」和训练侧的假量化
  逐位对比，只要对不上就报错退出。
- 第二条：把权重头文件打包成 Flash 常驻的 `infer/c/generated/nm_weights.c`。

---

## 质量验证闸门

```powershell
..\.venv\Scripts\python.exe -m pytest tests -q --ignore=tests\test_wkv7.py   # 52 passed
..\.venv\Scripts\python.exe -m pytest tests\test_wkv7.py -q                  # 10 passed
```

*`test_wkv7.py` 要单独跑：它导入时会 JIT 编译 WKV7 的 CUDA 算子，其中 2 条用例需要 CUDA（没有就 skip）。*