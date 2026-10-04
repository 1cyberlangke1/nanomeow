/* nanomeow 的板级配置默认值（引脚 / 波特率）。
 *
 * 输入：无（编译期宏）。输出：NM_OLED_SCL_PIN / NM_OLED_SDA_PIN / NM_UART_BRR 三个宏。
 * 预期行为：引擎自带一份默认接线（PB6 = SCL、PB7 = SDA、115200）；工程里想让
 *           User/config.h 当唯一改线处时，用 -DNM_BOARD_CONFIG=\"config.h\" 指过去，
 *           本文件先把那个头文件读进来，下面每个 #ifndef 就不会覆盖工程里的值。
 */
#ifndef NANOMEOW_BOARD_H
#define NANOMEOW_BOARD_H

#ifdef NM_BOARD_CONFIG
#include NM_BOARD_CONFIG
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
