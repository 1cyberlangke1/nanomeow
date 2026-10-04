/* 自动生成，请勿手改：infer/c/gen_font.py */
#ifndef NANOMEOW_FONT_H
#define NANOMEOW_FONT_H

#include <stdint.h>

#define NM_FONT_N 357          /* 收录字符数 */
#define NM_FONT_CP_BYTES 400   /* 码点表字节数 */
#define NM_FONT_W 8           /* 点阵宽（含原字体的右侧字间距列） */
#define NM_FONT_H 8           /* 点阵高（含原字体的顶部空行） */
#define NM_FONT_ROW 7         /* 实际存的像素行数：原字体第 0 行恒空 */
#define NM_FONT_INK 7         /* 实际存的像素列数：原字体第 7 列恒空 */

/* 码点表：LEB128 增量流，首个是绝对码点，之后是与前一项的差，按码点升序。 */
extern const uint8_t nm_font_cp[NM_FONT_CP_BYTES];
/* 点阵：NM_FONT_N 个字形，每个 NM_FONT_ROW 行；行内 bit7..bit1 是第 0..6 列。 */
extern const uint8_t nm_font_bits[NM_FONT_N * NM_FONT_ROW];
/* 兜底字形：7x7 空心方框，字库里没有的码点用它。 */
extern const uint8_t nm_font_fallback[NM_FONT_ROW];

/* 输入：BMP 码点；输出：指向 NM_FONT_ROW 个行字节的常量指针，码点不在字库时返回兜底字形。
 * 预期行为：顺序扫描码点表（升序），超过目标码点立刻退出；只读常量，可重入。 */
const uint8_t *nm_font_lookup(uint32_t cp);

/* 输入：一个完整字符的 UTF-8 字节与字节数；输出：同 nm_font_lookup。
 * 预期行为：长度不够或序列非法时返回兜底字形，不读越界。 */
const uint8_t *nm_font_lookup_utf8(const uint8_t *s, int n);

#endif
