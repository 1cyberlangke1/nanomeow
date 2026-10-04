# infer — nanomeow C 语言定点推理引擎

这是一个跑在 **STM32F103C8T6**（Cortex-M3，64 KiB Flash / 20 KiB RAM）微控制器上的纯整数推理引擎：
权重全部量化成 INT8，WKV 递归状态用 INT32 存储，全程只做定点运算，不依赖任何硬件或软件浮点。

---

## 模块架构与源文件分布

| 模块路径 | 职责与功能说明 |
| :--- | :--- |
| `c/engine/nanomeow.c` | 前向网络调度骨架，实现 INT8 GEMV 线性层和 WKV7 递归状态更新 |
| `c/engine/nm_fixed.h` | 定点数学核心：Scale 因子用 int32 乘子加 int32 移位表示、银行家舍入法 (Round-to-even)、uint128 中间结果截断 |
| `c/engine/nm_gen.c` | 自回归贪心采样生成器，在整数域做重复惩罚控制，检测 `<ETX>` (0x03) 终止符 |
| `c/engine/nm_utf8.h` | 轻量的流式 UTF-8 状态机解码器（防止截断时吐出半个多字节字符） |
| `c/generated/nm_weights.c` | 常驻 Flash 的权重数据池与稀疏描述符（由 `tools/gen_weights.py` 生成） |
| `c/generated/nm_lut.h` | 非线性算子的 Q15 定点查表（exp 和 log1p 都走查表，由 `tools/gen_lut.py` 生成） |
| `c/generated/nm_font.c` | 把 8x8 点阵的汉字/ASCII 子集压成紧凑位流的字库（由 `tools/gen_font.py` 生成） |
| `c/display/nm_oled.c` | 管理 SSD1306 的 128x64 显存帧缓冲，做 2x 点阵字形缩放绘制，按脏页差分推送 |
| `c/display/nm_oled_port.c` | 硬件适配层：用 GPIO 模拟软件 I2C 的驱动 |
| `c/platform/nm_board.h` | 默认硬件引脚映射（默认 PB6=SCL、PB7=SDA；可用工程宏重定向覆盖） |
| `c/platform/nm_stm32f103.h` | 裸机寄存器地址定义和最简外设头文件 |
| `c/host/nm_chat.c` | 主机端 CLI 交互入口（可以从 stdin 传 prompt 进来测试） |
| `c/host/nm_bench.c` | 主机端前向推理的性能基准测试入口 |
| `c/selftest/` | C 语言模块的单元自检（覆盖定点运算、UTF-8、推理引擎、字库解码和 OLED 显示） |
| `ref/` | Python 金标定点参考实现（给 G1 逐位对拍闸门当比对基准） |
| `firmware/` | 交叉编译、链接脚本 (`m3.ld`)，以及基于 Unicorn 的 M3 指令级仿真器 (`oled_view.py`) |
| `tests/` | 自动化回归测试（G1 纯整数对拍一致性、G2 困惑度退化闸门） |

---

## 构建与运行

### 1. 权重导出与代码生成
重新训练拿到新权重后，要重新生成 C 语言数据源：
```powershell
.\.venv\Scripts\python.exe train\scripts\export_int8.py --ckpt train\out\<run>\sft.pth --out infer\model_weights.h
.\.venv\Scripts\python.exe infer\c\tools\gen_weights.py
```

### 2. 主机端编译与交互
用 CMake 构建主机端可执行程序：
```powershell
cmake -S infer\c -B build\infer -G Ninja
cmake --build build\infer

# 执行交互式前向测试（参数：最大生成长度、重复惩罚 Q16、惩罚窗口大小）
"user:你好`nbot:" | .\build\infer\nm_chat.exe 128 65536 0
```

不用 CMake 的话，也可以直接调标准 C99 编译器：
```powershell
gcc -std=c99 -O2 -Wall -Wextra -Werror -I infer/c -I infer/c/engine -I infer/c/generated -I infer/c/display -I infer/c/platform -o nm_chat.exe infer/c/host/nm_chat.c infer/c/engine/nm_gen.c infer/c/engine/nanomeow.c infer/c/generated/nm_weights.c infer/c/generated/nm_font.c
```

### 3. 固件交叉编译
用一键构建脚本跑交叉编译、符号检查和段尺寸统计：
```powershell
.\.venv\Scripts\python.exe infer\firmware\build_firmware.py
```
*该脚本构建完会自动扫描所有目标文件的未定义符号，确保链接产物里不出现任何硬件/软件浮点库调用（`__aeabi_f*` / `__aeabi_d*`）。*

---

## 质量验证闸门

所有代码改动都要过下面这套自动化测试：
```powershell
.\.venv\Scripts\python.exe -m pytest infer\tests -q
```
- **G1 闸门（逻辑一致性）**：在固定输入序列下，C 引擎每步前向输出的 logits 码和量化 scale，都要和 Python 金标定点参考 (`ref/`) **逐位严格一致**。
- **G2 闸门（精度退化控制）**：纯定点 INT8 引擎相对同架构 BF16 浮点前向的困惑度退化，严格限制在 **≤ 5%** 以内（实测只退化 0.20%）。

---

## 存储占用（Cortex-M3，Clang -Oz 编译）

各模块编译后的段分布明细（由 `build_firmware.py` 从最终 ELF 里实测得到）：

| 目标文件 (.o) | .text (代码) | .rodata (常量) | .bss (RAM) | 模块说明 |
| :--- | :--- | :--- | :--- | :--- |
| `nanomeow.o` | 9,146 B | 128 B | 13,329 B | 前向引擎与 WKV7 核心算子 |
| `nm_gen.o` | 660 B | 0 B | 256 B | 贪心生成与状态控制 |
| `nm_weights.o` | 0 B | 47,007 B | 0 B | 常驻 Flash 的紧凑权重池 |
| `nm_font.o` | 380 B | 5,246 B | 0 B | 730 字点阵位流与索引 |
| `nm_oled.o` | 1,176 B | 25 B | 1,036 B | 显存缓冲与排版管理 |
| `nm_oled_port.o`| 286 B | 0 B | 0 B | 软件 I2C 总线模拟 |
| `main.o` | 698 B | 43 B | 3,904 B | 系统初始化与串口回显 |

### 整机合计

| 物理存储类型 | 实测占用 | 硬件规格上限 | 剩余空间 | 占用率 |
| :--- | :--- | :--- | :--- | :--- |
| **Flash (ROM)** | **65,187 B (63.66 KiB)** | 65,536 B (64 KiB) | **349 B** | 99.47% |
| **SRAM (RAM)** | **18,532 B (18.10 KiB)** | 20,480 B (20 KiB) | **1,948 B** | 90.49% |

### 核心空间优化策略
1. **参数池无损紧凑编码**：Scale 尾数归一化后截断到 23 位，移位量按张量基线加行内微量偏移来编码；权重码按整行去重后打包进统一数组 `nm_pool[]`，把 51,057 B（49.9 KiB）的原始参数压到 46,135 B（45.1 KiB），常驻 Flash。
2. **零层死参数剪枝**：RWKV-7 第一层在结构上就不参与 Value 残差更新，对应的 `v0/v1/v2` 参数不用导出，省下 452 字节 Flash。
3. **点阵字库位流压缩**：730 个 8x8 字符剥掉空白边界后，用紧凑位流存储（每字只占 49 位），点阵加 LEB128 索引表总共只用 5,246 字节。

---

## 运行性能

### 1. 指令开销热点分布
通过指令行映射分析，单个 Token 走完一次前向，总共约执行 250 万条指令：

| 算子函数 | 每 Token 指令数 | 占比 | 功能说明 |
| :--- | :--- | :--- | :--- |
| `nm_requant_code_u128` | 501,239 | 20.1% | 对逐行输出做 128 位定点再量化 |
| `nm_u128_shr_round` | 308,787 | 12.4% | 128 位银行家舍入右移 |
| `nm_to_fixed` | 250,711 | 10.0% | 把张量量化码转成 Q20 定点表示 |
| `nm_quantize_dynamic` | 210,286 | 8.4% | Per-tensor 动态极值搜索与量化 |
| `nm_linear` | 209,304 | 8.4% | INT8 GEMV 矩阵乘向量的内积计算 |
| `nm_round_div` | 186,781 | 7.5% | 64 位定点除法 |
| `nm_forward_token` | 173,365 | 6.9% | 前向分发框架 |
| 其余函数 | 658,410 | 26.3% | 位流读取、归一化和其他辅助运算 |

*性能瓶颈主要在 64/128 位定点运算和再量化链路上（约占 46%），而不是传统大模型常见的 GEMV 乘加计算。*

### 2. 实测推理吞吐
- **Cortex-M3 周期模拟器 (`oled_view.py`)**：单个 Token 前向仿真吞吐约 **20.9 tps**。
- **STM32F103C8T6 实体开发板 (@72 MHz)**：算上 OLED 双倍放大刷屏和串口 115200 波特率回显的开销，实机吞吐稳定在 **1.6 tps** 左右。

---

## 字库覆盖与渲染兜底

- 字库共收录 **730 个字符**（95 个可打印 ASCII，加上 635 个按 SFT 语料出现频次挑出来的高频汉字），覆盖训练语料里 98.72% 的实际字符出现次数；
- **兜底方框渲染机制**：遇到字库没收的字符或模型异常输出时，OLED 驱动统一调用 `nm_font_fallback` 渲染成 7x7 空心矩形框，保证显示驱动能正常往下推移，不会死锁。
