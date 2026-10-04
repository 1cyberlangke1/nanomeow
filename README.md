# nanomeow

运行在 **STM32F103C8T6**（Cortex-M3 @ 72 MHz，64 KiB Flash / 20 KiB RAM）上的 **INT8 纯整数** 超微型语言模型：
自训练的极低维 RWKV-7 x070 架构，零浮点依赖的 C 语言轻量推理引擎，以及 SSD1306 OLED / 串口交互终端。

> **关于模型定位的说明**：
> 本模型只有 4.2 万（42K）参数，受单片机存储和容量限制，本质上就是在封闭语料上做**高强度微调过拟合（SFT），把特定句式背成固定回复**（小参数模型的“背题”），没有通用语言泛化推理能力。

---

## 模型与系统规格

| 维度 | 参数 / 规格 | 说明 |
| :--- | :--- | :--- |
| **基础架构** | RWKV-7 x070 | 3 层 / `n_embd=32` / `head_size=8` / `n_head=4` / `dim_ffn=32`，总参数量 42,912 |
| **分词与上下文** | Byte-level Tokenizer (V=256) | 无 BPE，纯字节级编解码，最大上下文长度 512 字节 |
| **提示词模板** | `user:<输入>\nbot:<输出>` | 冒号后无空格，序列结束符用 ASCII `<ETX>` (`0x03`) |
| **训练与量化** | QAT 对称 INT8 假量化 | 基于 PyTorch (`torch.compile` + CUDA JIT)，全参数定点训练 |
| **推理算力** | 不依赖浮点符号 | 权重 INT8、激活 Per-tensor 动态 INT8、WKV State INT32，目标文件无任何 `__aeabi_f*` / `__aeabi_d*` |
| **硬件终端** | STM32F103C8T6 | 串口 115200 8N1 收发指令，I2C 驱动 128x64 OLED 双倍放大显示，状态栏常驻实时 TPS |

---

## 整体流程

```
train/data/chitchat_para.jsonl       预置场景对话语料（封闭对话问答集）
        │   train/notebooks/prepare_dataset.ipynb（断点续传清洗流）
        ▼
train/dataset/nana_clean.jsonl       清洗后的 SFT 数据（单字段 text，模板 user:<问>\nbot:<答>）
        │   python -m src.train --stage sft（QAT 默认开）
        ▼
train/out/<run>/*.pth                训练权重 Checkpoint（含量化 Scale 参数）
        │   train/scripts/export_int8.py
        ▼
infer/model_weights.h                导出权重头文件：INT8 权值码 + Per-channel Scale 乘子与移位
infer/model_cfg.h                    导出结构配置常量
        │   infer/c/tools/gen_weights.py（字库与 LUT 查表分别由 gen_font.py / gen_lut.py 生成）
        ▼
infer/c/generated/nm_weights.c       Flash 常驻无损压缩权重池（nm_pool[]）
        │
        ├─ build/infer/nm_chat.exe   主机推理 CLI（与 infer/ref/ 纯 Python 定点参考逐位对拍一致）
        └─ keil_demo/ 或固件构建脚本  →  烧录部署至 STM32F103C8T6
```

---

## 目录结构

```
nanomeow/
├─ train/                 训练侧（Python / PyTorch）
│  ├─ data/               合成语料资源
│  ├─ notebooks/          prepare_dataset.ipynb：语料清洗与统计体检
│  ├─ src/                模型前向、QAT 量化注入、WKV7 递推与训练调度
│  │  └─ wkv7/            WKV7 算子实现（含纯 PyTorch 分块与 CUDA C++ 加速路径）
│  ├─ scripts/            export_int8.py：量化权重导出转换工具
│  └─ tests/              训练与算子单元测试
├─ infer/                 推理侧（C99 纯整数引擎）
│  ├─ c/
│  │  ├─ engine/          定点前向推理内核、WKV7 递推与自回归生成控制
│  │  ├─ generated/       自动生成的权重池、8x8 点阵字库、非线性查表（LUT）
│  │  ├─ display/         SSD1306 驱动层与软件 I2C 模拟
│  │  ├─ platform/        板级寄存器与引脚映射
│  │  ├─ host/            主机端交互与性能评测入口（nm_chat / nm_bench）
│  │  ├─ tools/           权重压缩、字库压缩打包工具脚本
│  │  └─ selftest/        C 语言引擎自检用例（定点、字库、OLED、UTF-8）
│  ├─ ref/                Python 定点金标参考实现（用于 G1 逐位对拍闸门）
│  ├─ firmware/           交叉编译链、链接脚本与基于 Unicorn 的 Cortex-M3 模拟器
│  ├─ font/               8x8 点阵字模源文件、词频选字与打包脚本
│  └─ tests/              对拍自动化测试（G1 逻辑等价性与 G2 困惑度退化闸门）
├─ keil_demo/             STM32 Keil MDK-ARM 工程（Project.uvprojx）
├─ LICENSE                Apache License 2.0
└─ NOTICE                 依赖项目与开源字形授权溯源声明
```

---

## 快速上手

### 1. 环境准备

推荐使用 Python 3.11、MSVC 2022 与 CUDA Toolkit 12.x。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install triton-windows numpy pytest unicorn ipykernel
```

### 2. 数据准备与模型训练

在 Jupyter 或 VS Code 中运行 `train/notebooks/prepare_dataset.ipynb` 清洗语料，产出
`train/dataset/nana_clean.jsonl`，随后启动 SFT：

```powershell
cd train
..\.venv\Scripts\python.exe -m src.train --stage sft
```

训练设置与产物见 [train/README.md](train/README.md)。

### 3. 权重导出与主机端推理验证

```powershell
# 导出权重并生成 C 代码
.\.venv\Scripts\python.exe train\scripts\export_int8.py --ckpt train\out\sft\sft.pth --out infer\model_weights.h
.\.venv\Scripts\python.exe infer\c\tools\gen_weights.py

# 编译主机推理程序
cmake -S infer\c -B build\infer -G Ninja
cmake --build build\infer

# 执行推理交互（PowerShell 示例）
"user:你好`nbot:" | .\build\infer\nm_chat.exe 128 65536 0
```

### 4. 固件构建与上板部署

固件可以用两条独立的工具链编译，两者最终二进制的实际尺寸都在真机上量过：

- **自动化构建脚本（Clang + LLD 跨平台交叉编译）**：
  ```powershell
  .\.venv\Scripts\python.exe infer\firmware\build_firmware.py
  ```
- **Keil MDK 工程**：
  直接打开 `keil_demo\Project.uvprojx`，按 `Build` (F7)，再用 ST-Link `Download` (F8)。

硬件接线引脚定义与串口通信协议详见 [keil_demo/README.md](keil_demo/README.md)。

---

## 资源占用与性能实测

Cortex-M3（STM32F103C8T6）的存储和算力都很吃紧，下面逐项实测：

### 1. 存储占用

| 工具链 | 编译选项 | Flash (上限 64 KiB) | SRAM (上限 20 KiB) | 编译状态 |
| :--- | :--- | :--- | :--- | :--- |
| **Keil MDK-ARM** | armclang 6.7, -Oz, microlib, LTO | **65,432 B (63.90 KiB)**<br/>*剩余 104 B* | **19,552 B (19.09 KiB)**<br/>*剩余 928 B* | 0 Error / 0 Warning |
| **Clang 工具链** | clang 19, -Oz, lld | **65,187 B (63.66 KiB)**<br/>*剩余 349 B* | **18,532 B (18.10 KiB)**<br/>*剩余 1,948 B* | 0 Error / 0 Warning |

*注：权重经基线差分、动态行去重和紧凑位流打包（nm_pool），从 51,057 B（49.9 KiB）原始参数压到 46,135 B（45.1 KiB）常驻 Flash；RAM 里只常驻三层 WKV 递归状态（各 1,124 B，INT32）和执行工作区。*

### 2. 推理吞吐与耗时实测

- **Cortex-M3 周期精确模拟器 (`infer/firmware/oled_view.py`)**：单 Token 前向模拟吞吐约为 **20.9 tps**。
- **STM32F103C8T6 实机 (@72 MHz)**：在串口驱动和 OLED 双倍字号同步刷新下，实测吞吐稳定在 **1.6 tps**。

---

## 开源项目致谢

nanomeow 用到了下面这些开源项目（均按各自原始协议授权，详见 `NOTICE`）：

- **架构与算子**：[Mini_RWKV_7](https://github.com/Alic-Li/Mini_RWKV_7) (Apache-2.0) —— 参考了它的 RWKV-7 x070 架构和核心状态更新算子。
- **量化方案**：[pytorch/ao](https://github.com/pytorch/ao) (BSD-3-Clause) —— 量化感知训练（QAT）中的对称 INT8 假量化设计。
- **定点数值规范**：[CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) (Apache-2.0) —— 定点乘子、移位及饱和计算规范。
- **点阵字模**：[fusion-pixel-font](https://github.com/TakWolf/fusion-pixel-font) 与 [MisekiBitmap](https://github.com/ItMarki/MisekiBitmap) (SIL OFL 1.1) —— 提供嵌入式 OLED 屏所用的精简 8x8 汉字点阵字库。

---

## 许可证

本项目依据 **Apache License 2.0** 许可证开源，完整条款见 `LICENSE`。
