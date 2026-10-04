# infer — RWKV-7 x070 缩维模型的 INT8 纯整数推理

目标平台 STM32F103C8T6（Cortex-M3，64 KiB Flash / 20 KiB RAM）。权重全 int8、state 存 int32、
不用浮点；字节级 tokenizer（V=256，模板 `user:<内容>\nbot:<内容>`，ctx_len=512）；
权重由训练侧 QAT 产物导出，导出脚本会自检「反量化 == 训练侧假量化」逐位一致。

## 文件

| 路径 | 内容 |
|---|---|
| `c/engine/nanomeow.c` | 前向与 wkv7 递推（纯整数） |
| `c/engine/nm_fixed.h` | 定点算术：scale 的 (int32 乘子, int8 移位) 表示、四舍六入五成双、u128 中间量 |
| `c/generated/nm_lut.h` | exp / log1p 的 Q15 查表 |
| `c/engine/nm_gen.c` | 贪心生成、整数重复惩罚、`<ETX>`(0x03) 停止 |
| `c/engine/nm_utf8.h` | 增量 UTF-8 解码，不吐半截序列 |
| `c/generated/nm_weights.c` | 权重描述表，由 `gen_weights.py` 生成 |
| `c/generated/nm_font.c` / `c/generated/nm_font.h` | 8x8 子集字库与查表，由 `gen_font.py` 生成 |
| `c/display/nm_oled.c` / `c/display/nm_oled.h` | SSD1306 128x64 显示层：帧缓冲、8x8 字形渲染、按脏页刷新；最后一页留给固定状态行（tps） |
| `c/display/nm_oled_port.c` / `c/platform/nm_stm32f103.h` | 上板移植层：软件 I2C 与最小寄存器表（引脚从 `nm_board.h` 取，默认 PB6/PB7） |
| `c/platform/nm_board.h` | 板级默认值（SCL / SDA 引脚、USART1 的 BRR）；工程定义 `NM_BOARD_CONFIG` 时先读工程那份 |
| `keil_demo/User/main.c` | 上板固件入口：72 MHz 时钟 + USART1 收行 + OLED 显示（正文区 + 右下角常驻 tps）；clang 链路自带向量表 |
| `keil_demo/` | Keil MDK 工程：`Project.uvprojx`、`Start/startup_nanomeow.s`、`nanomeow.sct`、`User/config.h`（唯一改线处） |
| `firmware/` | 上板构建：`m3.ld`（64K/20K 链接脚本）、`build_firmware.py`（交叉编译 + 段账本 + 闸门）、`arm_inc/`（freestanding 下的 `string.h` 桩） |
| `font/` | 字库来源说明、选字脚本 `select_subset.py`、选字清单与许可原件 |
| `c/host/nm_bench.c` | 主机吞吐基准：只量 `nm_forward_token` 的周期/token（CMake 目标 `nm_bench`） |
| `firmware/arm_instcount.py` | 目标 ISA 静态账本：用上板同一套 CFLAGS 编成汇编，数每函数指令条数 |
| `firmware/arm_linecost.py` | 目标 ISA 动态账本：gcov 行执行次数 x clang `.loc` 行指令数，估每 token 每函数指令条数 |
| `firmware/oled_view.py` | OLED 可视化模拟：在 Cortex-M3 模拟器里真跑固件，把软件 I2C 位流还原成屏上字符画 |
| `c/selftest/*_selftest.c` | 定点内核 / UTF-8 解码器 / 引擎 / 字库 / OLED 五个自检 |
| `model_weights.h`、`model_cfg.h` | 导出产物：int8 码 + 每条 scale 的 (乘子, 移位) |
| `ref/` | Python 定点参考实现，C 引擎与它逐位对拍 |
| `tests/` | G1（逐位一致）/ G2（困惑度退化）闸门 |

## 重新导出权重

```powershell
.\.venv\Scripts\python.exe train\scripts\export_int8.py --ckpt train\out\<run>\sft.pth --out infer\model_weights.h
.\.venv\Scripts\python.exe infer\c\tools\gen_weights.py
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
gcc -std=c99 -O2 -Wall -Wextra -Werror -I infer/c -I infer/c/engine -I infer/c/generated -I infer/c/display -I infer/c/platform -o nm_chat.exe infer/c/host/nm_chat.c infer/c/engine/nm_gen.c infer/c/engine/nanomeow.c infer/c/generated/nm_weights.c infer/c/generated/nm_font.c
```

上板入口是 `keil_demo/User/main.c`：它把系统时钟配到 72 MHz（HSE 8 MHz × 9）、开 USART1
（PA9/PA10，115200 8N1）、点 SSD1306，然后循环「显示 `user:` → 收一行 → 拼
`user:<内容>\nbot:` → 生成 → 边生成边把字符刷到屏上」，同一份字节也回显到串口，每轮生成完
在屏上显示这次的 tps。OLED 走软件 I2C（开漏，~400 kHz），**引脚和波特率只改
`keil_demo/User/config.h`**（工程用 `-DNM_BOARD_CONFIG="config.h"` 把它交给 `nm_board.h`；
引擎自带默认 PB6 = SCL、PB7 = SDA）。两条链路共用这份入口：Keil 侧向量表由
`keil_demo/Start/startup_nanomeow.s` 提供（宏 `NM_VECTORS_IN_STARTUP`），clang 侧没有 startup
文件，由 `main.c` 自带向量表与 `Reset_Handler`。只想自己接管显示的话，实现一个
`nm_oled_bus_write` 就能复用整个显示层。

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

账本由上面那条 `build_firmware.py` 打出（clang 链路，向量表在 `main.c` 里，不靠 `-u`）。

| 目标文件 | .text | .rodata | .bss（RAM） |
|---|---|---|---|
| nanomeow.o（前向 + wkv7 + 定点底座） | 9,238 | 128 | 9,545 |
| nm_gen.o（贪心生成 / 重复惩罚 / 停止） | 800 | 0 | 1,280 |
| nm_weights.o（权重池 + 描述符） | 0 | 47,007 | 0 |
| nm_font.o（8x8 字库） | 380 | 5,038 | 0 |
| nm_oled.o（SSD1306 显示层） | 1,064 | 25 | 1,036 |
| nm_oled_port.o（软件 I2C） | 286 | 0 | 0 |
| main.o（时钟 / 串口 / 主循环 + 向量表） | 666 | 43 | 3,904 |

整机（链接后按程序头统计，含启动代码、`memcpy` / `memset` 与 64 位整数辅助函数）：

| 段 | 字节 |
|---|---|
| .isr_vector | 192 |
| .text | 12,638 |
| .rodata | 52,436 |
| **Flash 合计** | **65,266 = 63.74 KiB（64 KiB 余 270 B）** |
| .bss（RAM） | 15,772 = 15.40 KiB（20 KiB 余 4,708 B） |

同一条链路用 **Keil MDK（armclang 6.7 + microlib + LTO + scatter `keil_demo/nanomeow.sct`）**
编出来（`UV4 -b keil_demo/Project.uvprojx`，0 Error / 0 Warning）：

| 项 | 字节 |
|---|---|
| Code | 12,956 |
| RO-data | 52,480 |
| **Flash 合计** | **65,436 = 63.90 KiB（64 KiB 余 116 B）** |
| RW-data + ZI-data（RAM） | 16,280 = 15.90 KiB（20 KiB 余 4,200 B） |

两条链路是同一份权重、同一套源码：Keil 侧比 clang 侧多 170 B（microlib 的启动与库辅助代码
`memcpy` / `memset` / 64 位除法与移位），余量因此从 270 B 降到 116 B；这 170 B 由「第 0 层
`v0/v1/v2` 不导出」腾出的 452 B（见下节）覆盖。

无浮点复核查的是**各目标文件的未定义符号**，不是链接产物 —— lld 出来的 ELF 没有符号表，
在它上面查永远是「无」。实测只有 `memcpy` / `memset` 与整数辅助
`__aeabi_ldivmod / uldivmod / llsl / llsr / lasr`，没有 `__aeabi_f* / __aeabi_d*`。

**权重表是无损压缩后存的**（`gen_weights.py` 编码，`nanomeow.c` 的 `nm_row_scale` /
`nm_row_codes` 解码，两边口径互逆）：① scale 的尾数只存 23 位（归一化后最高位恒 1），
移位按张量存 1 字节基线 + 行内小位宽增量；② 权重码整张量行去重，只存唯一行池 + 每行索引；
③ 所有张量的码与 scale 位流首尾相接拼成一个 `nm_pool[]`，描述符只存 uint16 偏移（12 B → 8 B）。
三项把权重从 49,428 B 压到 46,587 B。`infer/tests/test_weights_pack.py` 会把池反解回来与
`model_weights.h` 的原始 mul / shift / 码逐位对拍，不需要 checkpoint。

**另有 452 B 是引擎永远读不到的**：第 0 层不做 value 残差（`nanomeow.c` 的 `if (layer == 0)`
直接拷 `v_first`，参考实现 `Mini_RWKV_7` 同样跳过），它的 `blocks.0.att.v0 / v1 / v2`
（36 + 58 + 358 B）`gen_weights.py` 直接不导出、描述符写 `{0, 0, 0, 0, 0, 0}`。这是**删掉
读不到的数据**，不是砍架构 —— `test_weights_pack.py` 有闸门守着「只有这三个允许留空」。

**权重码没有再压的余地（实测，可复现）**：整块 `nm_pool[]` 46,587 B 的字节熵是 7.809 bit/byte
（均匀 8.0），理论下限只到 45,477 B（省 2.4%）；拿通用压缩器压整块，`zlib -9` 得 43,560 B（6.50%）、
`lzma` 得 43,052 B（7.59%）。省下的这 3.5 KB 要靠解压换，而前向每个 token 都会把所有张量读一遍，
等于每 token 多跑一次全池解压 —— 收益远小于代价，所以 41 KB 的 int8 码按原样存。
（复现：把 `nm_weights.c` 里 `nm_pool` 的字节抠出来量熵 / 丢给 zlib 即可，不需要 checkpoint。）

**字库是 730 字**：可打印 ASCII 95 个无条件收录，中文按语料词频取到字节上限（730 - 95 = 635 个，
语料字符覆盖 98.72%）。点阵每字只存 7 行 x 7 列（原字体第 0 行与第 7 列恒空）= 49 位，
再紧密打包成位流，所以 730 字只占 4,473 B；码点表 766 B；查表代码 380 B。取舍见 `font/README.md`。

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

**主机口径**（`c/host/nm_bench.c`，只调 `nm_forward_token`，不碰生成路径）：用 `QueryThreadCycleTime`
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

字库现在覆盖语料字符的 **98.72%**（按出现次数；挤掉的是 66 个最冷门的字）。上面这张演示表里就有
8 个字（`亲住埋怀掖角诶跑`）落在字库外，OLED 上会画兜底方框；模型跑飞时吐的语料外字符
（`✉䜈丆享兌含坸戩攀槠聜裉蹸躬辜酌龼`）同样落到它身上 —— 兜底字形无论如何都要有。
