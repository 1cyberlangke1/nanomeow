# keil_demo — STM32F103C8T6 上板工程

本项目用 Keil MDK-ARM (ARMCLANG v6) 构建，把 nanomeow 纯整数模型跑在 **STM32F103C8T6**（Cortex-M3，72 MHz，64 KiB Flash / 20 KiB RAM）上。

上电后系统自动配置时钟和外设，从 USART1 读入一行 UTF-8 提示词，交给 RWKV-7 INT8 纯定点引擎做自回归推理，生成结果边算边输出到 SSD1306 128x64 OLED 屏和串口终端；屏幕状态行一直显示实测推理 TPS。

---

## 硬件外设与引脚配置

| 外设接口 | MCU 引脚 | 信号电气特性 | 功能说明 |
| :--- | :--- | :--- | :--- |
| **SSD1306 SCL** | `PB8` | 开漏输出 (Open-Drain) | 软件 I2C 时钟线 (~400 kHz) |
| **SSD1306 SDA** | `PB9` | 开漏输出 / 输入读取 | 软件 I2C 数据线（读取 IDR 接收 ACK） |
| **USART1 TX** | `PA9` | 复用推挽输出 | 串口输出（115200 8N1，回显生成文本） |
| **USART1 RX** | `PA10` | 浮空输入 | 串口输入（接收用户输入的提示词） |
| **供电电源** | 3.3V / GND | 直流电源 | 屏幕模块与 USB-TTL 需与 MCU 共地 |

> **改引脚接法**：
> 要换 I2C 引脚或串口波特率，改 `keil_demo/User/config.h` 就行（工程用宏 `-DNM_BOARD_CONFIG="config.h"` 把这份配置注入底层驱动）。引擎默认用 PB6/PB7，`config.h` 里覆盖成了实际接线的 PB8/PB9。两个 I2C 引脚必须在同一个 GPIOB 端口上。

---

## 工程目录结构

```
keil_demo/
├─ Project.uvprojx          Keil MDK 工程配置文件（ARMCLANG 6.7、microlib、LTO 开启）
├─ nanomeow.sct             分散加载脚本（精确划定 Flash 0x08000000 与 SRAM 布局）
├─ Start/
│  └─ startup_nanomeow.s    汇编启动文件（中断向量表、Reset_Handler、1024 字节轻量栈）
├─ User/
│  ├─ main.c                上板主循环入口（外设初始化、USART 接收缓冲、流式前向与 OLED 渲染）
│  └─ config.h              板级引脚与波特率重定义头文件
├─ DebugConfig/             调试适配器配置
├─ Objects/                 编译中间及最终目标（Project.axf、Project.hex）
└─ Listings/                编译链接内存映射清单（Project.map）
```

*注：工程源码直接引用上级目录 `infer/c/` 下的核心组件（`engine/`、`generated/`、`display/`），不用再复制一份代码。*

---

## 编译与固件烧录

### 1. Keil MDK 编译
1. 使用 Keil MDK-ARM 打开 `keil_demo/Project.uvprojx`。
2. 按 `Project -> Build Target`（快捷键 F7）。
3. 构建结果是 **0 Error / 0 Warning**，生成 `Objects/Project.hex` 和 `Project.axf`。

### 2. 固件下载
- **ST-Link（推荐）**：工程里已经配好 `STM32F10x_128.FLM` 算法，接上 SWD 接口（SWCLK、SWDIO、3V3、GND）后，直接点 Keil 的 `Download`（F8）就能烧录。
- **串口 ISP**：把开发板 BOOT0 置 1、BOOT1 置 0，复位后进入内置 Bootloader，再用 STM32CubeProgrammer 或 FlyMcu 通过串口把 `Project.hex` 写进芯片 Flash。烧完把 BOOT0 拨回 0 并复位。

---

## 串口交互协议与显示逻辑

- **通信参数**：115200 波特率，8 位数据位，无校验位（8N1），换行以 `\n` 为准（`\r` 自动忽略）。
- **交互流程**：
  1. 启动时 OLED 屏幕居中显示 `nanomeow / ready` 欢迎语；
  2. 收到一行有效输入后，先清掉上一屏内容，在第一行显示输入文本；
  3. 按 `user:<输入>\nbot:` 拼好提示词交给模型，一边生成一边把字符画到屏幕，同时从串口回显；
  4. 遇到终止标记 `<ETX>` 或到 128 字节上限就停下，屏幕右下角刷新显示这一轮的实测 TPS。

---

## 存储占用

数据取自 `keil_demo/Listings/Project.map` 的最终链接统计：

| 内存段分类 | 实际占用字节 | 硬件规格上限 | 空间剩余 | 占用率 |
| :--- | :--- | :--- | :--- | :--- |
| **Code** (代码段) | 12,936 B | - | - | - |
| **RO-data** (只读常量) | 52,496 B | - | - | - |
| **Flash 合计 (Code + RO)** | **65,432 B (63.90 KiB)** | 65,536 B (64 KiB) | **104 B** | 99.84% |
| **SRAM 合计 (RW + ZI)** | **19,552 B (19.09 KiB)** | 20,480 B (20 KiB) | **928 B** | 95.46% |

- **栈空间优化**：模型前向计算用的大块暂存都收拢到了静态共享缓冲区（`.bss`），`startup_nanomeow.s` 里最大物理栈深度压到 1,024 字节（静态分析出的峰值用量是 768 B，安全余量充足）。
- **实板推理吞吐**：STM32F103 跑在标准 72 MHz 主频下，实测端到端稳定吞吐约 **1.6 tps**。

---

## 运行注意事项

1. **点阵放大倍率**：默认开 2x 像素扩散放大（`NM_OLED_SCALE = 2`），单个字渲染成 16x16 像素，屏幕每行放 8 个字。
2. **状态栏隔离**：屏幕一共 8 页（Page 0~7），正文占 Page 0~5，最后两页（Page 6~7）单独作状态栏显示实时 TPS，正文往上滚时不会把状态栏冲掉。
3. **字模缺失时的表现**：字库收录 730 个常用字符，碰到字模表里没有的字，屏幕上一律显示成 7x7 像素的空心方框。
