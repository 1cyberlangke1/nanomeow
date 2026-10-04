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

## 资源账本（Cortex-M3，clang -Oz）

| 目标文件 | 段 | 字节 |
|---|---|---|
| nanomeow.o | .text 8,984 / .rodata 1,028 / .ARM.exidx 440 | 10,452 |
| nm_gen.o | .text 800 / .ARM.exidx 16 | 816 |
| nm_weights.o | .rodata 47,900 | 47,900 |
| 合计 Flash | | 59,168 = 57.8 KiB |
| 合计 RAM（.bss） | | 9,800 = 9.6 KiB |

引擎之外还得放：启动代码、中断向量表、memcpy/memset，以及 64 位整数辅助函数
（__aeabi_ldivmod、__aeabi_uldivmod、__aeabi_llsl/llsr/lasr —— 引擎里用了 int64 除法与移位）。
这些用等价替身实现量过：.text 404 + .ARM.exidx 72 + .isr_vector 192 + .rodata 4 = 672 B。
一起算上，整机约 59,840 B = 58.4 KiB，64 KiB 余约 5.8 KB。

**权重表是无损压缩后存的**（`gen_weights.py` 编码，`nanomeow.c` 的 `nm_row_scale` /
`nm_row_codes` 解码，两边口径互逆）：① scale 的尾数只存 23 位（归一化后最高位恒 1），
移位按张量存 1 字节基线 + 行内小位宽增量；② 权重码整张量行去重，只存唯一行池 + 每行索引。
两项把权重表从 49,428 B 压到 46,586 B（省 2,842 B），代价是解码代码多约 190 B，
链接层面净省 2,648 B。`infer/tests/test_weights_pack.py` 会把位流反解回来与
`model_weights.h` 的原始 mul / shift / 码逐位对拍，不需要 checkpoint。

**真实链接实测**（msys2 clang 22.1.8 用 `--target=armv7m-none-eabi -mthumb -mcpu=cortex-m3 -Oz
-ffreestanding -fno-unwind-tables` 编代码，再用 zig 自带的 lld 链接，链接脚本就是
STM32F103C8T6 的 64K/20K）：

| 段 | 字节 |
|---|---|
| .isr_vector | 192 |
| .text | 9,772 |
| .rodata | 48,932 |
| .ARM.exidx | 512 |
| **Flash 合计** | **59,408 = 58.02 KiB（64 KiB 余 6,128 B）** |
| .bss（RAM） | 12,148 = 11.86 KiB（20 KiB 余 8,332 B） |

链接**成功**，不是估算；替身运行时（复位入口 + 中断向量表 + 一次前向）已经算在 .text 里。
无浮点复核：`objdump -t` 看目标文件的未定义符号，只有 `memcpy/memset` 与整数辅助
`__aeabi_ldivmod / uldivmod / llsl / llsr / lasr`，**没有 `__aeabi_f* / __aeabi_d*`，也没有
`__aeabi_lmul`**。

两点结论：

- 16×16 汉字库（13,760 B）塞不进去，所以上板改用 **8x8 子集字库**：`font/` 下 422 字、
  点阵+码点表 3,412 B、含查表代码合计 3,756 B。全量 8x8（698 字 / 5,618 B）比现在多
  2,206 B，取舍见 `font/README.md`。
- OLED 驱动与固件入口已经写完（`c/nm_oled.c`、`c/nm_oled_port.c`、`c/nm_fw.c`），用仓库里的
  `firmware/build_firmware.py` 真编真链（入口就是 `nm_fw.c` 的 `Reset_Handler`，不靠 `-u`）：
  .isr_vector 192 + .text 12,686 + .rodata 52,416 = **65,294 B = 63.76 KiB（64 KiB 余 242 B）**；
  .bss **14,736 B = 14.39 KiB（20 KiB 余 5,744 B，剩下的正好当栈）**。
  `firmware/m3.ld` 里丢弃了 `.ARM.exidx`（裸机没有异常处理，只有调试器回溯会用到），
  白捡的 512 B 全给了字库：357 字 → 422 字。242 B 只够这个固件本身，再往上加东西
  （按键、更大字库、更多显示）就得先腾 Flash。
  工具链会影响结果：zig 自己的代码生成明显差（实测 zig -Oz 单目标
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

> 注：上面的 tok/s 是**墙钟**口径，机器一被别的进程抢核就失真（实测同一二进制两次能差 14%）。
> 之后改用 `tmp/_bench2.c` 的 `QueryThreadCycleTime`，只数本线程真正跑掉的周期，抖动 < 1%：
> 当前 **≈ 347K cycles/token**（`nm_range_error == 0`）。逐项优化与消融的结论见 `findings.md`
> 的「第十七轮性能账本」：采样 profile 把 29.6% 归给 `nm_quantize_dynamic`，但**消融否掉了**
> 这个归因 —— 逐元素取码的除法只占 1.0%、整表拷贝只占 0.2%，热点是分散的。
> ARM 侧静态账本（clang -Oz 的 Thumb 指令条数）：`nm_read_bits` 20、`nm_row_scale` 42、
> `nm_row_codes` 27、`nm_quantize_dynamic` 157、`nm_linear` 110、`nm_forward_token` 1,338。
