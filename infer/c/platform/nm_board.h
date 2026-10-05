/* nanomeow 的板级配置默认值（串口 / 引脚 / 波特率）。
 *
 * 输入：无（编译期宏）。输出：NM_UART_PORT / NM_OLED_SCL_PIN / NM_OLED_SDA_PIN / NM_UART_BRR 四个宏。
 * 预期行为：引擎自带一份默认接线（USART1、PB6 = SCL、PB7 = SDA、115200）；工程里想让
 *           User/config.h 当唯一改线处时，用 -DNM_BOARD_CONFIG=\"config.h\" 指过去，
 *           本文件先把那个头文件读进来，下面每个 #ifndef 就不会覆盖工程里的值。
 */
#ifndef NANOMEOW_BOARD_H
#define NANOMEOW_BOARD_H

#ifdef NM_BOARD_CONFIG
#include NM_BOARD_CONFIG
#endif

/* 串口走哪一路 USART：1 = USART1（PA9/PA10）、2 = USART2（PA2/PA3）、3 = USART3（PB10/PB11）。
 * 引脚、时钟使能位、寄存器基址都由这个值在 main.c 的 nm_uart_init 里收敛，换口不用改别处。 */
#ifndef NM_UART_PORT
#define NM_UART_PORT 1
#endif

#ifndef NM_OLED_SCL_PIN
#define NM_OLED_SCL_PIN 6
#endif
#ifndef NM_OLED_SDA_PIN
#define NM_OLED_SDA_PIN 7
#endif
#ifndef NM_UART_BRR
#define NM_UART_BRR 0x271u   /* 72 MHz 下 115200：39.0625 → 尾数 39、小数 1 */
#endif

#endif
