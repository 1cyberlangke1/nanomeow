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
gcc -std=c99 -O2 -Wall -Wextra -Werror -I infer/c -o nm_chat.exe infer/c/nm_chat.c infer/c/nm_gen.c infer/c/nanomeow.c infer/c/nm_weights.c
```

上板：把 c/ 下的 .c/.h（不含 nm_chat.c）加进工程，头文件路径指向 infer/c 与 infer/；
调用顺序是 nm_reset → 逐 token nm_forward_token → nm_gen_pick 取下一个字节。

## 闸门

```powershell
.\.venv\Scripts\python.exe -m pytest infer\tests -q
```

- G1：同一串 token 下，C 引擎与 ref/ 每一步的 logits 码与 scale 逐位相同。
- G2：int8 引擎相对同权重 bf16 前向的困惑度退化 ≤5%（实测 0.20%）。

## 资源账本（Cortex-M3，clang -Oz）

| 目标文件 | 段 | 字节 |
|---|---|---|
| nanomeow.o | .text 8,220 / .rodata 1,028 / .ARM.exidx 392 | 9,640 |
| nm_gen.o | .text 800 / .ARM.exidx 16 | 816 |
| nm_weights.o | .rodata 53,856 | 53,856 |
| 合计 Flash | | 64,312 = 62.8 KiB |
| 合计 RAM（.bss） | | 9,800 = 9.6 KiB |

引擎之外还得放：启动代码、中断向量表、memcpy/memset，以及 64 位整数辅助函数
（__aeabi_ldivmod、__aeabi_uldivmod、__aeabi_llsl/llsr/lasr —— 引擎里用了 int64 除法与移位）。
这些用等价替身实现量过：.text 404 + .ARM.exidx 72 + .isr_vector 192 + .rodata 4 = 672 B。
一起算上，整机约 64,984 B = 63.46 KiB，64 KiB 只剩 552 B 余量。

**真实链接实测**（msys2 clang 22.1.8 用 `--target=armv7m-none-eabi -mthumb -mcpu=cortex-m3 -Oz
-ffreestanding -fno-unwind-tables` 编代码，再用 zig 自带的 lld 链接，链接脚本就是
STM32F103C8T6 的 64K/20K）：

| 段 | 字节 |
|---|---|
| .isr_vector | 192 |
| .text | 9,624 |
| .rodata | 54,888 |
| .ARM.exidx | 480 |
| **Flash 合计** | **65,184 = 63.66 KiB（64 KiB 余 352 B）** |
| .bss（RAM） | 12,148 = 11.86 KiB（20 KiB 余 8,332 B） |

链接**成功**，不是估算；替身运行时（复位入口 + 中断向量表 + 一次前向）已经算在 .text 里。
无浮点复核：`objdump -t` 看目标文件的未定义符号，只有 `memcpy/memset` 与整数辅助
`__aeabi_ldivmod / uldivmod / llsl / llsr / lasr`，**没有 `__aeabi_f* / __aeabi_d*`，也没有
`__aeabi_lmul`**。

两点结论：

- 16×16 汉字库（13,760 B）塞不进去，上板显示中文要另想办法（外部 Flash / 更小字库）。
- 余量只有 352 B。工具链会影响结果：zig 自己的代码生成明显差（实测 zig -Oz 单目标
  14,190 B vs clang -Oz 9,640 B），所以本机用「clang 编 + zig 的 lld 链」而不是 `zig cc`
  一键编链；上板前最好再用 arm-none-eabi-gcc 复核一次。

## 主机吞吐实测

`tmp/_bench.c` 只调 `nm_forward_token`（不碰生成路径），在 RTX4060 笔记本的 i9-12900HX 上
用 `gcc -O2` 编：

| 版本 | tok/s | ms/token |
|---|---|---|
| 优化前 | 4,279.6 | 0.2337 |
| 优化后（CLZ 内建 / 2 的幂除法走移位 / requant 8 位长除法 + 64 位快路 / 乘法溢出快路 / GEMV 输入降位） | **8,264.5** | **0.1210** |

**1.93x**，全程 `nm_range_error == 0`，`pytest infer/tests` 56 passed、`pytest train/tests` 62 passed。
`gcov` 数出来的每 token 次数：requant 4360、码->Q16 rescale 8084、GEMV 乘加 33106、动态量化 236。
