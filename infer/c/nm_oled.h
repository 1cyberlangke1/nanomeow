/* nanomeow 的 SSD1306（128x64，I2C）显示层：帧缓冲 + 8x8 子集字库渲染。
 *
 * 输入：调用方给的字符（UTF-8 字节或码点），以及一个「把字节段发给屏」的传输钩子。
 * 输出：无返回值；屏幕内容由 nm_oled_flush 按脏页刷出去。
 * 预期行为：帧缓冲是 8 页 x 128 列，正好对齐 SSD1306 的页寻址；字库的 7 行点阵画在页内
 *           bit1..bit7（字库第 0 行恒空、第 7 列恒空），所以一行 8 像素 = 16 个字。
 *           传输钩子由移植层实现（上板是软件 I2C，主机端测试是假总线），本文件不含任何
 *           硬件寄存器、不含浮点。
 */
#ifndef NANOMEOW_OLED_H
#define NANOMEOW_OLED_H

#include <stdint.h>

#define NM_OLED_W 128                  /* 屏宽（像素） */
#define NM_OLED_H 64                   /* 屏高（像素） */
#define NM_OLED_PAGES (NM_OLED_H / 8)  /* 8 页，一页 = 一行字 */
#define NM_OLED_COLS (NM_OLED_W / 8)   /* 一行 16 个字 */

/* SSD1306 的 I2C 从地址（7 位 0x3C 左移一位）。 */
#define NM_OLED_I2C_ADDR 0x78u

/* 传输钩子：ctrl = 0x00 是命令流，0x40 是显示数据流（SSD1306 的 Co=0、D/C# 位）。
 * 预期行为：把 n 个字节按 SSD1306 的 I2C 帧发出去；上板由 nm_oled_port.c 实现，主机端测试用假总线捕获。 */
void nm_oled_bus_write(uint8_t ctrl, const uint8_t *p, int n);

/* 输入：无；输出：无。预期行为：发初始化序列 → 清屏 → 整屏刷新一次。 */
void nm_oled_init(void);

/* 输入：无；输出：无。预期行为：清帧缓冲并把光标回到左上；要显示得再调 nm_oled_flush。 */
void nm_oled_clear(void);

/* 输入：无；输出：无。预期行为：把「上次刷新后被改过」的页写进屏，没脏的页不发。 */
void nm_oled_flush(void);

/* 输入：无；输出：无。预期行为：光标回到左上角（不清屏、不刷屏）。 */
void nm_oled_home(void);

/* 输入：无；输出：无。预期行为：换到下一行；已经在最后一行则整屏上滚一页。 */
void nm_oled_newline(void);

/* 输入：一个码点；输出：无。预期行为：'\n' 换行，其余控制字符不占格，可打印字符画字库
 *           字形（字库外给兜底方框）并右移一格，行尾自动换行。 */
void nm_oled_putc(uint32_t cp);

/* 输入：一个字符的 UTF-8 字节与字节数（1..4）；输出：无。
 * 预期行为：与 nm_font_glyph_utf8 同口径解码；非法或半截序列画兜底方框。 */
void nm_oled_put_utf8(const uint8_t *b, int n);

/* 输入：任意字节流与长度；输出：无。预期行为：内部做增量 UTF-8 解码，只画完整字符，
 *           结尾没凑齐的半截序列直接丢掉（上板时用来回显串口收到的一行）。 */
void nm_oled_puts(const uint8_t *s, int n);

#endif