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

两点结论：

- 16×16 汉字库（13,760 B）塞不进去，上板显示中文要另想办法（外部 Flash / 更小字库）。
- 余量只有 0.5 KiB，而本机没有 arm-none-eabi-gcc，也没有 ARM 版 lld（mingw 的 ld 不支持
  armelf），所以上面的数字是「引擎逐段 + 替身运行时」的估算，不是真实链接结果；
  换 arm-none-eabi-gcc 会有出入，上板前需要用真实工具链复核一次。
