# infer — RWKV-7 x070 缩维模型的 INT8 纯整数推理

目标平台 STM32F103C8T6（Cortex-M3，64 KiB Flash / 20 KiB RAM）。权重全 int8、state 存 int32、
不用浮点；字节级 tokenizer（V=256，模板 `user:<内容>\nbot:<内容>`，ctx_len=512）；
权重由训练侧 QAT 产物导出，导出脚本会自检「反量化 == 训练侧假量化」逐位一致。

## 文件

| 路径 | 内容 |
|---|---|
| `c/nanomeow.c` | 前向与 wkv7 递推（纯整数） |
| `c/nm_fixed.h` | 定点算术：scale 的 (int32 乘子, int8 移位) 表示、四舍六入五成双、u128 中间量 |
| `c/nm_lut.h` | exp / log1p 的 Q15 查表 |
| `c/nm_gen.c` | 贪心生成、整数重复惩罚、`<ETX>`(0x03) 停止 |
| `c/nm_utf8.h` | 增量 UTF-8 解码，不吐半截序列 |
| `c/nm_weights.c` | 权重描述表，由 `gen_weights.py` 生成 |
| `c/nm_font.c` / `c/nm_font.h` | 8x8 子集字库与查表，由 `gen_font.py` 生成 |
| `c/nm_oled.c` / `c/nm_oled.h` | SSD1306 128x64 显示层：帧缓冲、8x8 字形渲染、按脏页刷新 |
| `c/nm_oled_port.c` / `c/nm_stm32f103.h` | 上板移植层：PB6/PB7 软件 I2C 与最小寄存器表 |
| `c/nm_fw.c` | 上板固件入口：72 MHz 时钟 + USART1 收行 + OLED 显示 |
| `firmware/` | 上板构建：`m3.ld`（64K/20K 链接脚本）、`build_firmware.py`（交叉编译 + 段账本 + 闸门）、`arm_inc/`（freestanding 下的 `string.h` 桩） |
| `font/` | 字库来源说明、选字脚本 `select_subset.py`、选字清单与许可原件 |
| `c/nm_bench.c` | 主机吞吐基准：只量 `nm_forward_token` 的周期/token（CMake 目标 `nm_bench`） |
| `firmware/arm_instcount.py` | 目标 ISA 静态账本：用上板同一套 CFLAGS 编成汇编，数每函数指令条数 |
| `firmware/arm_linecost.py` | 目标 ISA 动态账本：gcov 行执行次数 x clang `.loc` 行指令数，估每 token 每函数指令条数 |
| `c/*_selftest.c` | 定点内核 / UTF-8 解码器 / 引擎对拍三个自检 |
| `model_weights.h`、`model_cfg.h` | 导出产物：int8 码 + 每条 scale 的 (乘子, 移位) |
| `ref/` | Python 定点参考实现，C 引擎与它逐位对拍 |
| `tests/` | G1（逐位一致）/ G2（困惑度退化）闸门 |

## 重新导出权重

```powershell
.\.venv\Scripts\python.exe train\scripts\export_int8.py --ckpt train\out\<run>\sft.pth --out infer\model_weights.h
.\.venv\Scripts\python.exe infer\c\gen_weights.py
```

## 构建与运行

```powershell
cmake -S infer\c -B build\infer -G Ninja
cmake --build build\infer
"user:你好`nbot:" | build\infer\nm_chat.exe 128 65536 0
```

`nm_chat` 的参数依次是生成长度上限、重复惩罚（Q16）、惩罚窗口；prompt 走 stdin，生成字节走 stdout。
不用 cmake 时等价的一行：

```powershell
gcc -std=c99 -O2 -Wall -Wextra -Werror -I infer/c -o nm_chat.exe infer/c/nm_chat.c infer/c/nm_gen.c infer/c/nanomeow.c infer/c/nm_weights.c infer/c/nm_font.c
```

上板：`c/nm_fw.c` 就是能直接烧的固件入口（自带复位向量与中断向量表），它把系统时钟配到
72 MHz（HSE 8 MHz × 9）、开 USART1（PA9/PA10，115200 8N1）、点 SSD1306，然后循环
「显示 `user:` → 收一行 → 拼 `user:<内容>\nbot:` → 生成 → 边生成边把字符刷到屏上」，
同一份字节也回显到串口。OLED 走 PB6 = SCL、PB7 = SDA 的软件 I2C（开漏，~400 kHz）。
把它和 `nanomeow.c` / `nm_gen.c` / `nm_weights.c` / `nm_font.c` / `nm_oled.c` / `nm_oled_port.c`
一起编进去即可；只想自己接管显示的话，实现一个 `nm_oled_bus_write` 就能复用整个显示层。

上板构建一条命令（链接脚本与闸门都在仓库里，交叉编译器默认取 PATH 上的 clang / zig，取不到再退回 msys2 ucrt64 与本机 ziglang）：

```powershell
.\.venv\Scripts\python.exe infer\firmware\build_firmware.py
```

它按程序头统计真正要烧进 Flash / RAM 的字节，并检查**各目标文件的未定义符号**里没有
`__aeabi_f*` / `__aeabi_d*`（纯整数）—— 不能在链接产物上查，lld 出来的 ELF 没有符号表，那样查永远是「无」。任一闸门不过就以非 0 退出。

## 闸门

```powershell
.\.venv\Scripts\python.exe -m pytest infer\tests -q
```

- G1：同一串 token 下，C 引擎与 ref/ 每一步的 logits 码与 scale 逐位相同。
- G2：int8 引擎相对同权重 bf16 前向的困惑度退化 ≤5%（实测 0.20%）。

## 资源账本（Cortex-M3，clang -Oz，真编译真链接）

上板入口是 `c/nm_fw.c` 的 `Reset_Handler`（自带中断向量表，不靠 `-u`），账本由上面那条
`build_firmware.py` 打出。

| 目标文件 | .text | .rodata | .bss（RAM） |
|---|---|---|---|
| nanomeow.o（前向 + wkv7 + 定点底座） | 9,238 | 128 | 9,545 |
| nm_gen.o（贪心生成 / 重复惩罚 / 停止） | 800 | 0 | 1,280 |
| nm_weights.o（权重池 + 描述符） | 0 | 47,459 | 0 |
| nm_font.o（8x8 字库） | 380 | 5,038 | 0 |
| nm_oled.o（SSD1306 显示层） | 1,010 | 25 | 1,036 |
| nm_oled_port.o（软件 I2C） | 266 | 0 | 0 |
| nm_fw.o（时钟 / 串口 / 主循环） | 434 | 42 | 3,900 |

整机（链接后按程序头统计，含启动代码、`memcpy` / `memset` 与 64 位整数辅助函数）：

| 段 | 字节 |
|---|---|
| .isr_vector | 192 |
| .text | 12,602 |
| .rodata | 52,696 |
| **Flash 合计** | **65,490 = 63.96 KiB（64 KiB 余 46 B）** |
| .bss（RAM） | 15,768 = 15.40 KiB（20 KiB 余 4,712 B） |

无浮点复核查的是**各目标文件的未定义符号**，不是链接产物 —— lld 出来的 ELF 没有符号表，
在它上面查永远是「无」。实测只有 `memcpy` / `memset` 与整数辅助
`__aeabi_ldivmod / uldivmod / llsl / llsr / lasr`，没有 `__aeabi_f* / __aeabi_d*`。

**权重表是无损压缩后存的**（`gen_weights.py` 编码，`nanomeow.c` 的 `nm_row_scale` /
`nm_row_codes` 解码，两边口径互逆）：① scale 的尾数只存 23 位（归一化后最高位恒 1），
移位按张量存 1 字节基线 + 行内小位宽增量；② 权重码整张量行去重，只存唯一行池 + 每行索引；
③ 所有张量的码与 scale 位流首尾相接拼成一个 `nm_pool[]`，描述符只存 uint16 偏移（12 B → 8 B）。
三项把权重从 49,428 B 压到 46,587 B。`infer/tests/test_weights_pack.py` 会把池反解回来与
`model_weights.h` 的原始 mul / shift / 码逐位对拍，不需要 checkpoint。

**权重码没有再压的余地（实测，可复现）**：整块 `nm_pool[]` 46,587 B 的字节熵是 7.809 bit/byte
（均匀 8.0），理论下限只到 45,477 B（省 2.4%）；拿通用压缩器压整块，`zlib -9` 得 43,560 B（6.50%）、
`lzma` 得 43,052 B（7.59%）。省下的这 3.5 KB 要靠解压换，而前向每个 token 都会把所有张量读一遍，
等于每 token 多跑一次全池解压 —— 收益远小于代价，所以 41 KB 的 int8 码按原样存。
（复现：把 `nm_weights.c` 里 `nm_pool` 的字节抠出来量熵 / 丢给 zlib 即可，不需要 checkpoint。）

**字库是全量 701 字**：语料 bot 侧的 701 个不同字符一个不少。点阵每字只存 7 行 x 7 列
（原字体第 0 行与第 7 列恒空）= 49 位，再紧密打包成位流，所以 701 字只占 4,295 B；
码点表 736 B；查表代码 380 B。取舍与方框率见 `font/README.md`。

工具链会影响结果：zig 自己的代码生成明显差（实测 zig -Oz 单目标 14,190 B vs clang -Oz 9,640 B），
所以本机用「clang 编 + zig 的 lld 链」而不是 `zig cc` 一键编链；上板前最好再用
arm-none-eabi-gcc 复核一次。

## 性能账本

三个口径互补，工具都在仓库里，clone 下来就能复现：

```powershell
cmake -S infer\c -B build\infer -G Ninja
cmake --build build\infer
build\infer\nm_bench.exe 20000                              # 主机口径：周期/token
.\.venv\Scripts\python.exe infer\firmware\arm_instcount.py   # 目标 ISA 静态口径：每函数指令条数
.\.venv\Scripts\python.exe infer\firmware\arm_linecost.py    # 目标 ISA 动态口径：每 token 每函数指令条数（估算）
```

**主机口径**（`c/nm_bench.c`，只调 `nm_forward_token`，不碰生成路径）：用 `QueryThreadCycleTime`
只数本线程真正跑掉的周期，同一二进制重复跑抖动 < 1%（墙钟口径实测两次能差 14%，已弃用）。
i9-12900HX 上实测三次 **308,972 / 312,639 / 309,588 cycles/token**（≈ 31 万），
全程 `nm_range_error == 0`（引擎的溢出计数，非 0 就说明中间量饱和、结果不可信）。
阶段 32 的基线是 347K，阶段 34 把 `nm_linear` 内层改成「两行共享 x 载入」后降到 31 万。

**目标 ISA 静态口径**（`firmware/arm_instcount.py`，用与上板**完全相同**的 CFLAGS 编成汇编再数指令行）：
7 个目标文件合计 4,292 条指令，最大一块是 `nm_forward_token` 1,350 条，其次
`nm_generate` 288、`nm_requant_code_u128` 191、`nm_linear` 160、`nm_quantize_dynamic` 157。

**目标 ISA 动态口径（估算）**（`firmware/arm_linecost.py`）：把「每行源码执行了几次」乘上
「每行源码编出几条指令」。执行次数来自主机 `gcc -O0 --coverage` 跑 `nm_bench`（引擎全是整数运算，
同一份源码在 Cortex-M3 上每行执行几次与主机逐行相同）；每行指令数来自 clang `-Oz -mcpu=cortex-m3 -g`
汇编里的 `.loc`。实测 **约 250 万条指令/token**：

| 函数 | 条/token | 占比 | 干什么 |
|---|---|---|---|
| `nm_requant_code_u128` | 501,239 | 20.1% | 逐行码的 128 位再量化（每 token 约 3,559 次调用） |
| `nm_u128_shr_round` | 308,787 | 12.4% | 上一条路径里的 128 位舍入右移 |
| `nm_to_fixed` | 250,711 | 10.0% | 张量码 -> Q20 定点（每 token 约 190 次 x 32 元素） |
| `nm_quantize_dynamic` | 210,286 | 8.4% | per-tensor 动态量化 |
| `nm_linear` | 209,304 | 8.4% | int8 GEMV（含点积内层） |
| `nm_round_div` | 186,781 | 7.5% | 四舍六入五成双的 64 位除法 |
| `nm_forward_token` | 173,365 | 6.9% | 前向骨架，含内联进来的那些 |
| `nm_read_bits` / `nm_normalize_u128` / `nm_mul_wide` / 其余 | 658,410 | 26.3% | |

按 1.3~1.8 CPI 折算 **约 325~450 万周期/token -> @72 MHz 约 45~62 ms/token（16~22 tok/s）**。

这张表的价值是指向下一步优化：**64 / 128 位定点运算那条链**（`nm_requant_code_u128` +
`nm_u128_shr_round` + `nm_round_div` + `nm_normalize_u128` + `nm_mul_wide`）合计约 115 万条/token、
占 **46%** —— 它是热点，不是 GEMV。

> 动态口径是**估算**，不是实测：行 -> 指令的映射会被内联 / 合并 / 消除影响，所以只能看量级与占比。
> 要给出「STM32 上到底多少 tok/s」，需要板子或周期精确的模拟器，仓库里目前**没有**这个口径的实测数据
> —— 上面三个都是间接口径，不能拿主机数字冒充目标机的 tok/s。

## 实测 good case

同一份 int8 权重在主机侧跑 `build/infer/nm_chat.exe`（PowerShell：`"user:<问>`nbot:" | build\infer\nm_chat.exe 128 65536 0`）：

| 问 | 答 |
|---|---|
| 你好 | 你好呀，奈奈在的，主人今天想聊点什么喵 |
| 晚安 | 那就躺下，奈奈给主人掖好被角，晚安喵 |
| 抱抱 | 被抱住啦，奈奈把脸埋进主人怀里蹭蹭喵 |
| 亲亲 | 诶，额头被亲了，奈奈脸好烫喵 |
| 奈奈是谁 | 奈奈是主人的猫娘小助手，会一直陪在主人身边喵 |
| 你叫什么名字 | 奈奈是主人的猫娘小助手，会一直陪在主人身边喵 |

演示 20 条问答的输出**零乱码、零方框**（字库现在是全量 701 字）；38 条演示/泛化输出共 735 字符里
还剩 17 个方框（2.31%），全是模型跑飞时吐的**语料外**字符（`✉䜈丆享兌含坸戩攀槠聜裉蹸躬辜酌龼`），
全量字库也覆盖不到，所以 OLED 上兜底方框无论如何都要有。
