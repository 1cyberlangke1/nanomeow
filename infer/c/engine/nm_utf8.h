/* nanomeow 增量 UTF-8 解码器：只吐完整字符，不吐半截序列、不吐 U+FFFD。
 *
 * 输入：逐个 feed 的字节（0..255）。
 * 输出：nm_utf8_push 把本次新解出的完整字符写进调用方缓冲，返回写出的字节数。
 * 预期行为：与 train/src/tokenizer.py 的 UTF8StreamDecoder 一致 —— 半截序列留在内部缓冲
 *           等后续字节；非法字节直接丢掉（丢的是出错序列的首字节，与 CPython 的报错位置一致）；
 *           nm_utf8_flush 丢掉结尾没凑齐的半截字节。缓冲区最多 4 字节，全部状态在结构体里，
 *           不依赖任何库、不含浮点，Cortex-M3 直接可用。
 */
#ifndef NANOMEOW_UTF8_H
#define NANOMEOW_UTF8_H

#include <stdint.h>

typedef struct {
    uint8_t buf[4];    /* 还没凑成完整字符的字节 */
    int len;           /* buf 里的有效字节数，恒 <= 4 */
    uint32_t dropped;  /* 累计丢掉的非法 / 半截字节数 */
} nm_utf8_decoder;

/* 输入：解码器；输出：无。预期行为：清空缓冲与计数。 */
static inline void nm_utf8_init(nm_utf8_decoder *d)
{
    d->len = 0;
    d->dropped = 0;
}

/* 输入：序列首字节；输出：该序列应有的长度（0 = 非法首字节）。
 * 预期行为：挡掉 0xC0 / 0xC1（过长编码）与 0xF5 以上（超出 U+10FFFF）。 */
static inline int nm_utf8_seq_len(uint8_t b)
{
    if (b < 0x80) return 1;
    if (b >= 0xC2 && b <= 0xDF) return 2;
    if (b >= 0xE0 && b <= 0xEF) return 3;
    if (b >= 0xF0 && b <= 0xF4) return 4;
    return 0;
}

/* 输入：候选序列、长度；输出：1 = 合法，0 = 非法。
 * 预期行为：续字节必须是 10xxxxxx，且解码值不能是过长编码、代理区或 > U+10FFFF。 */
static inline int nm_utf8_valid(const uint8_t *b, int n)
{
    uint32_t cp;
    int i;
    if (n == 1) return 1;
    for (i = 1; i < n; i++)
        if ((b[i] & 0xC0) != 0x80) return 0;
    if (n == 2) {
        cp = ((uint32_t)(b[0] & 0x1F) << 6) | (uint32_t)(b[1] & 0x3F);
        return cp >= 0x80;
    }
    if (n == 3) {
        cp = ((uint32_t)(b[0] & 0x0F) << 12) | ((uint32_t)(b[1] & 0x3F) << 6)
             | (uint32_t)(b[2] & 0x3F);
        return cp >= 0x800 && !(cp >= 0xD800 && cp <= 0xDFFF);
    }
    cp = ((uint32_t)(b[0] & 0x07) << 18) | ((uint32_t)(b[1] & 0x3F) << 12)
         | ((uint32_t)(b[2] & 0x3F) << 6) | (uint32_t)(b[3] & 0x3F);
    return cp >= 0x10000 && cp <= 0x10FFFF;
}

/* 输入：已经校验过的完整 UTF-8 序列、长度；输出：码点。
 * 预期行为：调用前必须先用 nm_utf8_seq_len 拿到长度、用 nm_utf8_valid 验过；这里不重复校验。
 *           抽出来是为了让 nm_font_glyph_utf8 与 nm_oled_put_utf8 共用一份解码，少一份 .text。 */
static inline uint32_t nm_utf8_decode(const uint8_t *b, int n)
{
    if (n == 1) return (uint32_t)b[0];
    if (n == 2) return ((uint32_t)(b[0] & 0x1F) << 6) | (uint32_t)(b[1] & 0x3F);
    if (n == 3)
        return ((uint32_t)(b[0] & 0x0F) << 12) | ((uint32_t)(b[1] & 0x3F) << 6)
               | (uint32_t)(b[2] & 0x3F);
    return ((uint32_t)(b[0] & 0x07) << 18) | ((uint32_t)(b[1] & 0x3F) << 12)
           | ((uint32_t)(b[2] & 0x3F) << 6) | (uint32_t)(b[3] & 0x3F);
}

/* 输入：序列缓冲、已经到手的字节数（1..need）；输出：1 = 目前还合法，0 = 现在就能判定非法。
 * 预期行为：CPython 的 UTF-8 解码器是边收边校验的，所以首字节的特殊范围
 *           （0xE0 的次字节 >= 0xA0、0xED 的次字节 <= 0x9F、0xF0 的次字节 >= 0x90、
 *           0xF4 的次字节 <= 0x8F，其余次字节必须是 10xxxxxx）在序列还没收齐时就要检查，
 *           这样「过长 / 代理区 / 越界」的半截序列才会在同一个位置被判非法。 */
static inline int nm_utf8_prefix_ok(const uint8_t *b, int have)
{
    if (have >= 2) {
        if (b[0] == 0xE0) { if (b[1] < 0xA0) return 0; }
        else if (b[0] == 0xED) { if (b[1] > 0x9F) return 0; }
        else if (b[0] == 0xF0) { if (b[1] < 0x90) return 0; }
        else if (b[0] == 0xF4) { if (b[1] > 0x8F) return 0; }
        else if ((b[1] & 0xC0) != 0x80) return 0;
    }
    if (have >= 3 && (b[2] & 0xC0) != 0x80) return 0;
    return 1;
}

/* 输入：解码器、一个字节、输出缓冲（至少 4 字节）；输出：写出的字节数。
 * 预期行为：凑齐一个合法字符就写出来（ASCII 立即写）；半截序列压住不写；
 *           已经能判定非法的序列（含续字节不是 10xxxxxx、过长、代理区、越界）立刻丢首字节，
 *           继续解后面的字节，绝不输出 U+FFFD。 */
static inline int nm_utf8_push(nm_utf8_decoder *d, uint8_t byte, uint8_t *out)
{
    int n = 0, i;
    d->buf[d->len++] = byte;
    for (;;) {
        int need, bad = 0;
        if (d->len == 0) break;
        need = nm_utf8_seq_len(d->buf[0]);
        if (need == 0) {
            bad = 1;                       /* 非法首字节 */
        } else if (d->len < need) {
            if (nm_utf8_prefix_ok(d->buf, d->len)) break;   /* 半截且目前合法：等后续字节 */
            bad = 1;
        } else if (!nm_utf8_valid(d->buf, need)) {
            bad = 1;                       /* 完整但非法（过长 / 代理区 / 越界） */
        }
        if (bad) {
            for (i = 1; i < d->len; i++) d->buf[i - 1] = d->buf[i];
            d->len--;
            d->dropped++;
            continue;
        }
        for (i = 0; i < need; i++) out[n + i] = d->buf[i];
        n += need;
        for (i = need; i < d->len; i++) d->buf[i - need] = d->buf[i];
        d->len -= need;
    }
    return n;
}

/* 输入：解码器；输出：丢掉的半截字节数。预期行为：缓冲清空，不产出任何字符。 */
static inline int nm_utf8_flush(nm_utf8_decoder *d)
{
    int n = d->len;
    d->dropped += (uint32_t)n;
    d->len = 0;
    return n;
}

#endif
