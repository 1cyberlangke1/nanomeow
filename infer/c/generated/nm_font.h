/* 自动生成，请勿手改：infer/c/tools/gen_font.py */
#ifndef NANOMEOW_FONT_H
#define NANOMEOW_FONT_H

#include <stdint.h>

#define NM_FONT_N 730          /* 收录字符数 */
#define NM_FONT_CP_BYTES 766   /* 码点表字节数 */
#define NM_FONT_PACKED_BYTES 4473   /* 点阵位流字节数（含表尾 1 个 0 补位） */
#define NM_FONT_W 8           /* 点阵宽（含原字体的右侧字间距列） */
#define NM_FONT_H 8           /* 点阵高（含原字体的顶部空行） */
#define NM_FONT_ROW 7         /* 实际存的像素行数：原字体第 0 行恒空 */
#define NM_FONT_INK 7         /* 实际存的像素列数：原字体第 7 列恒空 */
#define NM_FONT_GLYPH_BITS (NM_FONT_ROW * NM_FONT_INK)   /* 每字占的位数 */

/* 码点表：LEB128 增量流，首个是绝对码点，之后是与前一项的差，按码点升序。 */
extern const uint8_t nm_font_cp[NM_FONT_CP_BYTES];
/* 点阵位流：NM_FONT_N 个字，每个 NM_FONT_GLYPH_BITS 位；第 r 行 7 位（原字节 bit7..bit1）
 * MSB 在前。表尾多 1 个 0 字节，是解包时「固定读两个字节」的哨兵。 */
extern const uint8_t nm_font_packed[NM_FONT_PACKED_BYTES];
/* 兜底字形：7x7 空心方框，字库里没有的码点用它。 */
extern const uint8_t nm_font_fallback[NM_FONT_ROW];

/* 输入：BMP 码点、至少 NM_FONT_ROW 字节的输出缓冲；输出：无。
 * 预期行为：把该字形解包进 out（每行 bit7..bit1 = 第 0..6 列，bit0 恒 0），码点不在字库时写兜底；
 *           顺序扫描码点表（升序），超过目标码点立刻退出；只读常量，可重入。 */
void nm_font_glyph(uint32_t cp, uint8_t *out);

/* 输入：一个完整字符的 UTF-8 字节、字节数、输出缓冲；输出：无。
 * 预期行为：同 nm_font_glyph；长度不够或序列非法时写兜底字形，不读越界。 */
void nm_font_glyph_utf8(const uint8_t *s, int n, uint8_t *out);

#endif
