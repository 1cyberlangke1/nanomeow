#ifndef __CONFIG_H
#define __CONFIG_H

/* nanomeow 的板级接线：改线只改这一个文件。
 *
 * 输入：无（编译期宏）。输出：NM_UART_PORT / NM_OLED_SCL_PIN / NM_OLED_SDA_PIN / NM_UART_BRR。
 * 预期行为：Keil 工程里已经把本文件通过 -DNM_BOARD_CONFIG="config.h" 交给引擎
 *           （见 infer/c/platform/nm_board.h）；引擎自带的默认值是 USART1 + PB6/PB7，这里覆盖成实际接线。
 */

/* 串口走 USART2：TX = PA2、RX = PA3，两个脚都在 PA0~PA7 范围内。
 * 为什么只有这一对：STM32F103 的 USART1 = PA9/PA10、USART3 = PB10/PB11，
 * PA0~PA7 里带 USART 复用的只有 PA2/PA3，所以 8 个脚里能当硬件串口的就这一对。 */
#define NM_UART_PORT 2

/* SSD1306 走软件 I2C：SCL = PB8、SDA = PB9。两个脚必须同属 GPIOB。 */
#define NM_OLED_SCL_PIN 8
#define NM_OLED_SDA_PIN 9

/* USART2 的波特率寄存器值：USARTDIV = f_PCLK1 / (16 × 波特率)，BRR = USARTDIV × 16（4 位小数）。
 * USART2 挂 APB1，f_PCLK1 = 36 MHz，36 MHz / (16 × 115200) = 19.53125 → 尾数 19（0x13）、
 * 小数 0.53125 × 16 = 8.5 取 9 → 0x139（实际 19.5625，等效 115016 baud，误差 -0.16%）。
 * 换波特率就换这个值。 */
#define NM_UART_BRR 0x139u

#endif
