/* nanomeow 的 SSD1306 显示层实现：帧缓冲、初始化序列、8x8 字形放大渲染、按脏页刷新。
 *
 * 帧缓冲按 SSD1306 的页组织：一页 8 个像素行、128 列，共 8 页。字库是 8x8，上屏时每个
 * 像素铺成 NM_OLED_SCALE x NM_OLED_SCALE 个像素，于是一个字占 NM_OLED_CELL 列、一行
 * NM_OLED_COLS 个字。
 */
#include <string.h>

#include "nm_oled.h"
#include "nm_font.h"
#include "nm_utf8.h"

/* SSD1306 128x64 的上电序列：显示关 → 时钟 / 复用 / 偏移 / 起始行 → 电荷泵 → 水平寻址 →
 * 段重映射与 COM 扫描方向 → COM 引脚 / 对比度 / 预充电 / VCOMH → 显示跟随 RAM → 显示开。 */
static const uint8_t nm_init_seq[] = {
    0xAE,              /* 显示关 */
    0xD5, 0x80,        /* 时钟分频 1、振荡频率 8 */
    0xA8, 0x3F,        /* 复用比 64 */
    0xD3, 0x00,        /* 显示偏移 0 */
    0x40,              /* 起始行 0 */
    0x8D, 0x14,        /* 电荷泵开 */
    0x20, 0x00,        /* 水平寻址模式 */
    0xA1,              /* 段重映射：列 127 → SEG0 */
    0xC8,              /* COM 扫描反向：行 63 → COM0 */
    0xDA, 0x12,        /* COM 引脚配置（交替） */
    0x81, 0xCF,        /* 对比度 */
    0xD9, 0xF1,        /* 预充电周期 */
    0xDB, 0x40,        /* VCOMH 电平 */
    0xA4,              /* 显示跟随显示 RAM */
    0xA6,              /* 正常显示（不反白） */
    0xAF,              /* 显示开 */
};

static uint8_t nm_fb[NM_OLED_PAGES * NM_OLED_W];
static uint16_t nm_dirty;              /* bit p = 第 p 页需要重刷 */
static int nm_col;                     /* 光标列，单位是字（0..NM_OLED_COLS-1） */
static int nm_page;                    /* 光标页（0..7），一页 = 8 个像素行 */

#define NM_ALL_PAGES ((uint16_t)((1u << NM_OLED_PAGES) - 1u))

void nm_oled_clear(void)
{
    memset(nm_fb, 0, sizeof(nm_fb));
    nm_col = 0;
    nm_page = 0;
    nm_dirty = NM_ALL_PAGES;
}

void nm_oled_home(void)
{
    nm_col = 0;
    nm_page = 0;
}

/* 输入：无；输出：无。预期行为：**正文区**上移 NM_OLED_LINE_PAGES 页、正文区末尾
 * NM_OLED_LINE_PAGES 页清零，所有页标脏。
 * 状态行不在正文区里（最后 NM_OLED_LINE_PAGES 页留给右下角），所以它不会跟着滚。
 * 不用 memmove 是因为 Cortex-M3 的 freestanding 运行库只保证 memcpy / memset。 */
static void nm_oled_scroll(void)
{
    int i;
    const int step = NM_OLED_LINE_PAGES * NM_OLED_W;
    const int keep = (NM_OLED_TEXT_PAGES - NM_OLED_LINE_PAGES) * NM_OLED_W;

    for (i = 0; i < keep; i++)
        nm_fb[i] = nm_fb[i + step];
    memset(nm_fb + keep, 0, (size_t)step);
    nm_dirty = NM_ALL_PAGES;
}

void nm_oled_newline(void)
{
    nm_col = 0;
    nm_page += NM_OLED_LINE_PAGES;
    if (nm_page >= NM_OLED_TEXT_PAGES) {
        nm_oled_scroll();
        nm_page = NM_OLED_TEXT_PAGES - NM_OLED_LINE_PAGES;
    }
}

void nm_oled_goto(int page, int col)
{
    /* 页号上限再往下走 NM_OLED_LINE_PAGES 页，字块就整个画出屏外了。 */
    if (page >= 0 && page <= NM_OLED_PAGES - NM_OLED_LINE_PAGES
        && col >= 0 && col < NM_OLED_COLS) {
        nm_page = page;
        nm_col = col;
    }
}

/* 输入：7 行点阵；输出：无。预期行为：画在光标处 —— 字库第 r 行、第 c 列的点铺成
 * NM_OLED_SCALE x NM_OLED_SCALE 的实心方块，落在字块相对像素 (r*SCALE, c*SCALE)；
 * 字块从第 nm_page 页、第 nm_col*NM_OLED_CELL 列起，按页组织。标脏字块覆盖的那几页。 */
static void nm_oled_blit(const uint8_t *glyph)
{
    int r, c, dy, dx, k;

    for (r = 0; r < NM_FONT_ROW; r++) {
        uint8_t bits = glyph[r];
        if (!bits) continue;
        for (c = 0; c < 8; c++) {
            if (!(bits & (uint8_t)(0x80u >> c))) continue;
            for (dy = 0; dy < NM_OLED_SCALE; dy++) {
                int y = r * NM_OLED_SCALE + dy;
                uint8_t *dst = nm_fb + (nm_page + y / 8) * NM_OLED_W
                             + nm_col * NM_OLED_CELL + c * NM_OLED_SCALE;
                uint8_t bit = (uint8_t)(1u << (y % 8));

                for (dx = 0; dx < NM_OLED_SCALE; dx++)
                    dst[dx] |= bit;
            }
        }
    }
    for (k = 0; k < NM_OLED_LINE_PAGES; k++)
        nm_dirty |= (uint16_t)(1u << (nm_page + k));
}

void nm_oled_putc(uint32_t cp)
{
    uint8_t glyph[NM_FONT_ROW];

    if (cp == '\n') { nm_oled_newline(); return; }
    if (cp < 0x20u || cp == 0x7Fu) return;      /* 其余控制字符不占格 */
    nm_font_glyph(cp, glyph);
    nm_oled_blit(glyph);
    if (++nm_col < NM_OLED_COLS) return;
    /* 正文区行尾换行；状态行（最后 NM_OLED_LINE_PAGES 页）行尾停在最后一格 —— 走
     * nm_oled_newline 会触发正文区滚动，把正文（含用户输入那行）顶掉，tps 会吃掉它自己那行。 */
    if (nm_page >= NM_OLED_TEXT_PAGES) nm_col = NM_OLED_COLS - 1;
    else nm_oled_newline();
}

void nm_oled_put_utf8(const uint8_t *b, int n)
{
    int need = nm_utf8_seq_len(b[0]);

    if (need == 0 || n < need || !nm_utf8_valid(b, need)) {
        nm_oled_blit(nm_font_fallback);
        if (++nm_col >= NM_OLED_COLS) nm_oled_newline();
        return;
    }
    nm_oled_putc(nm_utf8_decode(b, need));
}

void nm_oled_puts(const uint8_t *s, int n)
{
    nm_utf8_decoder dec;
    uint8_t out[4];
    int i;

    nm_utf8_init(&dec);
    for (i = 0; i < n; i++) {
        int m = nm_utf8_push(&dec, s[i], out);
        if (m > 0) nm_oled_put_utf8(out, m);
    }
    nm_utf8_flush(&dec);      /* 结尾没凑齐的半截字符直接丢，不在屏上留兜底方块 */
}

void nm_oled_putu(uint32_t v)
{
    uint8_t buf[10];      /* uint32 最多 10 位十进制 */
    int n = 0;

    do { buf[n++] = (uint8_t)('0' + v % 10u); v /= 10u; } while (v != 0);
    while (n > 0) nm_oled_putc(buf[--n]);
}

void nm_oled_flush(void)
{
    int p;

    for (p = 0; p < NM_OLED_PAGES; p++) {
        uint8_t win[6];

        if (!(nm_dirty & (uint16_t)(1u << p))) continue;
        win[0] = 0x21; win[1] = 0x00; win[2] = NM_OLED_W - 1;      /* 列 0..127 */
        win[3] = 0x22; win[4] = (uint8_t)p; win[5] = (uint8_t)p;   /* 页 p..p */
        nm_oled_bus_write(0x00, win, 6);
        nm_oled_bus_write(0x40, nm_fb + p * NM_OLED_W, NM_OLED_W);
    }
    nm_dirty = 0;
}

void nm_oled_init(void)
{
    nm_oled_bus_write(0x00, nm_init_seq, (int)sizeof(nm_init_seq));
    nm_oled_clear();
    nm_oled_flush();
}
