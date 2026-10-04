/* 字库自检：码点表严格升序、每个字形都能解包、解包结果按同一口径重新打包后与位流逐位相同、
 * 字库外与非法 UTF-8 都给兜底字形。
 *
 * 输入：无（直接读 nm_font.c 里的常量表）。
 * 输出：全部通过打印一行 OK，任何一条不满足就打印失败项并以非 0 退出。
 * 预期行为：打包 / 解包只在这里互逆验证（重新打包那份用逐位写入独立写一遍，不是自洽比较）；
 *           点阵与 BDF 的逐位对拍在 infer/tests/test_font.py 里做（那边才有 BDF 原件与清单）。
 */
#include <stdio.h>
#include <string.h>

#include "nm_font.h"

static int failures;

static void check(int ok, const char *what, unsigned long arg)
{
    if (!ok) {
        printf("FAIL %s (%lu)\n", what, arg);
        failures++;
    }
}

/* 输入：7 行点阵、输出位流、起始位号；输出：无。
 * 预期行为：与 gen_font.py 的打包口径一致 —— 每行取 bit7..bit1 共 NM_FONT_INK 位、MSB 在前，
 *           第 r 行第 k 位对应点阵的 bit(7-k)。这里用逐位写入独立实现，和 nm_font_read7 的
 *           「16 位窗口右移」不是同一套写法，所以能验出位序错。 */
static void pack_glyph(const uint8_t *g, uint8_t *bits, uint32_t pos)
{
    int r, k;

    for (r = 0; r < NM_FONT_ROW; r++)
        for (k = 0; k < NM_FONT_INK; k++) {
            uint32_t bit = pos + (uint32_t)r * NM_FONT_INK + (uint32_t)k;
            if (g[r] & (uint8_t)(0x80u >> k))
                bits[bit >> 3] |= (uint8_t)(0x80u >> (bit & 7u));
        }
}

int main(void)
{
    uint32_t cps[NM_FONT_N];
    uint32_t prev = 0;
    int i, r, pos = 0;
    uint8_t glyph[NM_FONT_ROW];
    uint8_t first[NM_FONT_ROW];
    uint8_t again[NM_FONT_ROW];
    uint8_t packed[NM_FONT_PACKED_BYTES];
    uint8_t buf[4];
    int n = 0;

    for (i = 0; i < NM_FONT_N; i++) {
        uint32_t delta = 0;
        int shift = 0;
        uint8_t byte;
        do {
            if (pos >= NM_FONT_CP_BYTES) {
                printf("FAIL 码点表提前用尽\n");
                return 1;
            }
            byte = nm_font_cp[pos++];
            delta |= (uint32_t)(byte & 0x7F) << shift;
            shift += 7;
        } while (byte & 0x80);
        prev += delta;
        cps[i] = prev;
        check(i == 0 || cps[i] > cps[i - 1], "码点必须严格升序", (unsigned long)i);
    }
    check(pos == NM_FONT_CP_BYTES, "码点表必须正好用完", (unsigned long)pos);

    memset(packed, 0, sizeof(packed));
    for (i = 0; i < NM_FONT_N; i++) {
        nm_font_glyph(cps[i], glyph);
        for (r = 0; r < NM_FONT_ROW; r++)
            check((glyph[r] & 1u) == 0, "解包后第 7 列必须为 0", (unsigned long)i);
        pack_glyph(glyph, packed, (uint32_t)i * NM_FONT_GLYPH_BITS);
    }
    check(memcmp(packed, nm_font_packed, NM_FONT_PACKED_BYTES) == 0,
          "解包后重新打包必须与位流逐位相同", 0);

    nm_font_glyph(cps[0] - 1, glyph);
    check(memcmp(glyph, nm_font_fallback, NM_FONT_ROW) == 0, "更小的码点必须给兜底", 0);
    nm_font_glyph(cps[NM_FONT_N - 1] + 1, glyph);
    check(memcmp(glyph, nm_font_fallback, NM_FONT_ROW) == 0, "更大的码点必须给兜底", 0);
    nm_font_glyph(0xFFFFFFFFu, glyph);
    check(memcmp(glyph, nm_font_fallback, NM_FONT_ROW) == 0, "越界码点必须给兜底", 0);

    /* UTF-8 入口：把第一个码点编成 UTF-8，必须解出同一个字形；半截 / 非法序列给兜底 */
    if (cps[0] < 0x800) {
        buf[n++] = (uint8_t)(0xC0 | (cps[0] >> 6));
        buf[n++] = (uint8_t)(0x80 | (cps[0] & 0x3F));
    } else {
        buf[n++] = (uint8_t)(0xE0 | (cps[0] >> 12));
        buf[n++] = (uint8_t)(0x80 | ((cps[0] >> 6) & 0x3F));
        buf[n++] = (uint8_t)(0x80 | (cps[0] & 0x3F));
    }
    nm_font_glyph(cps[0], first);
    nm_font_glyph_utf8(buf, n, again);
    check(memcmp(again, first, NM_FONT_ROW) == 0, "UTF-8 入口必须解出同一个字形", 0);
    nm_font_glyph_utf8(buf, n - 1, again);
    check(memcmp(again, nm_font_fallback, NM_FONT_ROW) == 0, "半截 UTF-8 必须给兜底", 0);
    buf[0] = 0xFF;
    nm_font_glyph_utf8(buf, 1, again);
    check(memcmp(again, nm_font_fallback, NM_FONT_ROW) == 0, "非法 UTF-8 必须给兜底", 0);

    if (failures) {
        printf("%d 项失败\n", failures);
        return 1;
    }
    printf("OK %d 字 / 码点表 %d B / 点阵位流 %d B（原 %d B）\n",
           NM_FONT_N, NM_FONT_CP_BYTES, NM_FONT_PACKED_BYTES, NM_FONT_N * NM_FONT_ROW);
    return 0;
}