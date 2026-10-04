#ifndef __CONFIG_H
#define __CONFIG_H

/* nanomeow 的板级接线：改线只改这一个文件。
 *
 * 输入：无（编译期宏）。输出：NM_OLED_SCL_PIN / NM_OLED_SDA_PIN / NM_UART_BRR。
 * 预期行为：Keil 工程里已经把本文件通过 -DNM_BOARD_CONFIG="config.h" 交给引擎
 *           （见 infer/c/platform/nm_board.h）；引擎自带的默认值是 PB6/PB7，这里覆盖成实际接线。
 */

/* SSD1306 走软件 I2C：SCL = PB8、SDA = PB9。两个脚必须同属 GPIOB。 */
#define NM_OLED_SCL_PIN 8
#define NM_OLED_SDA_PIN 9

/* USART1 的波特率寄存器值：BRR = f_PCLK2 / (16 × 波特率)，72 MHz / 115200 = 39.0625
 * → 尾数 39（0x27）、小数 0.0625 × 16 = 1 → 0x271。换波特率就换这个值。 */
#define NM_UART_BRR 0x271u

#endif
