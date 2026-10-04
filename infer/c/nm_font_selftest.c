/* 字库自检：码点表严格升序、每个码点都查回自己的点阵、字库外与非法 UTF-8 都给兜底字形。
 *
 * 输入：无（直接读 nm_font.c 里的常量表）。
 * 输出：全部通过打印一行 OK，任何一条不满足就打印失败项并以非 0 退出。
 * 预期行为：只做结构 + 查表行为的检查，不依赖外部数据；点阵与 BDF 的逐位对拍在
 *           infer/tests/test_font.py 里做（那边才有 BDF 原件）。
 */
#include <stdio.h>

#include "nm_font.h"

static int failures;

static void check(int ok, const char *what, unsigned long arg)
{
    if (!ok) {
        printf("FAIL %s (%lu)\n", what, arg);
        failures++;
    }
}

int main(void)
{
    uint32_t cps[NM_FONT_N];
    uint32_t prev = 0;
    int i, pos = 0;
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

    for (i = 0; i < NM_FONT_N; i++)
        check(nm_font_lookup(cps[i]) == nm_font_bits + (size_t)i * NM_FONT_ROW,
              "码点必须查回自己的字形", (unsigned long)i);

    check(nm_font_lookup(cps[0] - 1) == nm_font_fallback, "更小的码点必须给兜底", 0);
    check(nm_font_lookup(cps[NM_FONT_N - 1] + 1) == nm_font_fallback, "更大的码点必须给兜底", 0);
    check(nm_font_lookup(0xFFFFFFFFu) == nm_font_fallback, "越界码点必须给兜底", 0);

    /* UTF-8 入口：把第一个码点编成 UTF-8，必须查到同一个字形；半截 / 非法序列给兜底 */
    if (cps[0] < 0x800) {
        buf[n++] = (uint8_t)(0xC0 | (cps[0] >> 6));
        buf[n++] = (uint8_t)(0x80 | (cps[0] & 0x3F));
    } else {
        buf[n++] = (uint8_t)(0xE0 | (cps[0] >> 12));
        buf[n++] = (uint8_t)(0x80 | ((cps[0] >> 6) & 0x3F));
        buf[n++] = (uint8_t)(0x80 | (cps[0] & 0x3F));
    }
    check(nm_font_lookup_utf8(buf, n) == nm_font_bits, "UTF-8 入口必须查回第一个字形", 0);
    check(nm_font_lookup_utf8(buf, n - 1) == nm_font_fallback, "半截 UTF-8 必须给兜底", 0);
    buf[0] = 0xFF;
    check(nm_font_lookup_utf8(buf, 1) == nm_font_fallback, "非法 UTF-8 必须给兜底", 0);

    if (failures) {
        printf("%d 项失败\n", failures);
        return 1;
    }
    printf("OK %d 字 / 码点表 %d B / 点阵 %d B\n",
           NM_FONT_N, NM_FONT_CP_BYTES, NM_FONT_N * NM_FONT_ROW);
    return 0;
}
