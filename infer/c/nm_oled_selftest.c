/* OLED 显示层自检：用假 I2C 总线捕获 nm_oled.c 发出去的字节流，按 SSD1306 的水平寻址规则
 * 重建显示 RAM，再逐条核对初始化序列、清屏、字形落点、换行、滚动、兜底与控制字符。
 *
 * 输入：无（只调 nm_oled 的公开接口，字库用 nm_font.c 的真实表）。
 * 输出：全部通过打印一行 OK，任何一条不满足就打印失败项并以非 0 退出。
 * 预期行为：不依赖硬件与外部数据；页窗口只可能是「列 0..127、页 p..p」，重建器据此收字节。
 */
#include <stdio.h>
#include <string.h>

#include "nm_font.h"
#include "nm_oled.h"

/* 与 nm_oled.c 的初始化序列逐字节一致的期望值：这里独立抄一份，实现被改坏时不会自洽通过。 */
static const uint8_t expect_init[] = {
    0xAE, 0xD5, 0x80, 0xA8, 0x3F, 0xD3, 0x00, 0x40, 0x8D, 0x14, 0x20, 0x00,
    0xA1, 0xC8, 0xDA, 0x12, 0x81, 0xCF, 0xD9, 0xF1, 0xDB, 0x40, 0xA4, 0xA6, 0xAF,
};

static uint8_t dram[NM_OLED_PAGES][NM_OLED_W];
static uint8_t init_frame[64];
static uint8_t last_win[6];
static int init_len, last_win_len, cur_col, cur_page, col_hi, page_hi;
static int n_cmd, n_data, last_data_len;

/* 输入：ctrl 与字节段（假总线入口）；输出：无。
 * 预期行为：数据帧按水平寻址依次填进显示 RAM；第一帧命令留档当初始化序列；其余命令只解析页窗口。 */
void nm_oled_bus_write(uint8_t ctrl, const uint8_t *p, int n)
{
    int i;

    if (ctrl == 0x40) {
        n_data++;
        last_data_len = n;
        for (i = 0; i < n; i++) {
            dram[cur_page][cur_col] = p[i];
            if (++cur_col > col_hi) {
                cur_col = 0;
                if (++cur_page > page_hi) cur_page = 0;
            }
        }
        return;
    }
    n_cmd++;
    if (init_len == 0) {
        init_len = n < (int)sizeof(init_frame) ? n : (int)sizeof(init_frame);
        memcpy(init_frame, p, (size_t)init_len);
        return;
    }
    if (n == 6 && p[0] == 0x21 && p[3] == 0x22) {
        last_win_len = n;
        memcpy(last_win, p, 6);
        cur_col = p[1]; col_hi = p[2];
        cur_page = p[4]; page_hi = p[5];
    }
}

static int failures;

static void check(int ok, const char *what, long arg)
{
    if (!ok) { printf("FAIL %s (%ld)\n", what, arg); failures++; }
}

/* 输入：7 行点阵；输出：8 个竖排字节。预期行为：字库第 r 行 → 页内 bit(r+1)，
 * 字库 bit(7-c) → 第 c 列，与 nm_oled_blit 同一口径。 */
static void expect_cell(const uint8_t *glyph, uint8_t *out)
{
    int r, c;

    memset(out, 0, 8);
    for (r = 0; r < NM_FONT_ROW; r++)
        for (c = 0; c < 8; c++)
            if (glyph[r] & (uint8_t)(0x80u >> c))
                out[c] |= (uint8_t)(1u << (r + 1));
}

static int cell_is(int page, int col, const uint8_t *glyph)
{
    uint8_t want[8];

    expect_cell(glyph, want);
    return memcmp(dram[page] + col * 8, want, 8) == 0;
}

/* 输入：页、列、BMP 码点；输出：1 表示该格正好是那个码点的字形。
 * 预期行为：先 nm_font_glyph 解包进栈上缓冲，再走 cell_is 的同一口径比较。 */
static int cell_is_cp(int page, int col, uint32_t cp)
{
    uint8_t glyph[NM_FONT_ROW];

    nm_font_glyph(cp, glyph);
    return cell_is(page, col, glyph);
}

/* 输入：BMP 码点；输出：1 表示该码点在字库里有真字形（不是兜底方框）。 */
static int in_font(uint32_t cp)
{
    uint8_t glyph[NM_FONT_ROW];

    nm_font_glyph(cp, glyph);
    return memcmp(glyph, nm_font_fallback, NM_FONT_ROW) != 0;
}

static int cell_is_zero(int page, int col)
{
    int i;

    for (i = 0; i < 8; i++)
        if (dram[page][col * 8 + i]) return 0;
    return 1;
}

static void reset_bus(void)
{
    memset(dram, 0, sizeof(dram));
    memset(last_win, 0, sizeof(last_win));
    init_len = 0; last_win_len = 0; n_cmd = 0; n_data = 0; last_data_len = 0;
    cur_col = 0; cur_page = 0; col_hi = NM_OLED_W - 1; page_hi = NM_OLED_PAGES - 1;
}

int main(void)
{
    uint8_t before[NM_OLED_PAGES][NM_OLED_W];
    int i, p, c;

    /* 1. 初始化：序列逐字节相同，然后清屏 + 整屏刷一遍 */
    reset_bus();
    nm_oled_init();
    check(init_len == (int)sizeof(expect_init), "初始化序列长度", init_len);
    check(init_len == (int)sizeof(expect_init)
          && memcmp(init_frame, expect_init, sizeof(expect_init)) == 0, "初始化序列字节", 0);
    check(n_cmd == 1 + NM_OLED_PAGES, "init 后应有 1 帧初始化 + 8 帧页窗口", n_cmd);
    check(n_data == NM_OLED_PAGES, "init 后应有 8 帧页数据", n_data);
    check(last_data_len == NM_OLED_W, "页数据帧长度", last_data_len);
    check(last_win_len == 6 && last_win[0] == 0x21 && last_win[1] == 0x00
          && last_win[2] == NM_OLED_W - 1 && last_win[3] == 0x22
          && last_win[4] == last_win[5], "页窗口命令必须是「列 0..127、页 p..p」", 0);
    for (p = 0; p < NM_OLED_PAGES; p++)
        for (c = 0; c < NM_OLED_W; c++)
            check(dram[p][c] == 0, "init 后显示 RAM 必须全 0", p * NM_OLED_W + c);

    /* 2. 一个字符：只刷一页，落点在 (0, 0) */
    reset_bus();
    nm_oled_putc('A');
    nm_oled_flush();
    check(n_cmd == 1 && n_data == 1, "单字符只应刷一页", n_cmd * 10 + n_data);
    check(cell_is_cp(0, 0, 'A'), "'A' 的落点", 0);
    for (c = 1; c < NM_OLED_COLS; c++)
        check(cell_is_zero(0, c), "'A' 不应动到别的格子", c);

    /* 3. 一行 16 个字，第 17 个换行到 (1, 0) */
    reset_bus();
    nm_oled_clear();
    for (i = 0; i < NM_OLED_COLS; i++) nm_oled_putc((uint32_t)('a' + i));
    nm_oled_putc('q');
    nm_oled_flush();
    check(cell_is_cp(1, 0, 'q'), "第 17 个字应换行到 (1,0)", 0);
    check(cell_is_cp(0, 0, 'a'), "第一个字应留在 (0,0)", 0);

    /* 4. UTF-8 入口：「你好」两个三字节字符，都要用真字形（不是兜底） */
    reset_bus();
    nm_oled_clear();
    nm_oled_puts((const uint8_t *)"\xE4\xBD\xA0\xE5\xA5\xBD", 6);
    nm_oled_flush();
    check(in_font(0x4F60u), "「你」必须在字库里", 0);
    check(in_font(0x597Du), "「好」必须在字库里", 0);
    check(cell_is_cp(0, 0, 0x4F60u), "「你」的落点", 0);
    check(cell_is_cp(0, 1, 0x597Du), "「好」的落点", 0);

    /* 5. 字库外的码点与半截 UTF-8：都画兜底方框 */
    reset_bus();
    nm_oled_clear();
    nm_oled_putc(0xE000u);
    nm_oled_flush();
    check(cell_is(0, 0, nm_font_fallback), "字库外码点应画兜底", 0);

    reset_bus();
    nm_oled_clear();
    nm_oled_put_utf8((const uint8_t *)"\xE4\xBD", 2);
    nm_oled_flush();
    check(cell_is(0, 0, nm_font_fallback), "半截 UTF-8 应画兜底", 0);

    /* 6. puts 结尾的半截字符直接丢，不留兜底方块 */
    reset_bus();
    nm_oled_clear();
    nm_oled_puts((const uint8_t *)"\xE4\xBD", 2);
    nm_oled_flush();
    check(cell_is_zero(0, 0), "puts 的半截结尾不应画任何东西", 0);

    /* 7. 控制字符不占格、不产生任何绘制 */
    reset_bus();
    memcpy(before, dram, sizeof(dram));
    nm_oled_putc(0x01u);
    nm_oled_putc('\r');
    nm_oled_putc(0x7Fu);
    nm_oled_flush();
    check(n_data == 0, "控制字符不应产生任何绘制", n_data);
    check(memcmp(dram, before, sizeof(dram)) == 0, "控制字符不应改动显示 RAM", 0);
    nm_oled_putc('A');
    nm_oled_flush();
    check(cell_is_cp(0, 0, 'A'), "控制字符后光标仍应停在 (0,0)", 0);

    /* 8. 走到屏幕底部再换行：整屏上滚一页、末页清零 */
    reset_bus();
    nm_oled_clear();
    for (i = 0; i < 7 * NM_OLED_COLS + 1; i++)
        nm_oled_putc((uint32_t)('A' + i % 26));
    nm_oled_flush();
    check(cell_is_cp(7, 0, (uint32_t)('A' + (7 * NM_OLED_COLS) % 26)), "第 113 个字应落在 (7,0)", 0);
    memcpy(before, dram, sizeof(dram));
    nm_oled_newline();
    nm_oled_flush();
    for (p = 0; p < NM_OLED_PAGES - 1; p++)
        check(memcmp(dram[p], before[p + 1], NM_OLED_W) == 0, "上滚后每页应等于原下一页", p);
    for (c = 0; c < NM_OLED_W; c++)
        check(dram[NM_OLED_PAGES - 1][c] == 0, "上滚后末页必须清零", c);

    if (failures) {
        printf("%d 项失败\n", failures);
        return 1;
    }
    printf("OK 初始化序列 %d B / %d 页 / 一行 %d 字 / 帧缓冲 %d B\n",
           (int)sizeof(expect_init), NM_OLED_PAGES, NM_OLED_COLS, NM_OLED_PAGES * NM_OLED_W);
    return 0;
}