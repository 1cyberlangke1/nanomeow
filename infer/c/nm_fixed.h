/* nanomeow 定点底座：scale 表示与全部整数算式（与 infer/ref/fixed.py 逐位一致）。
 *
 * scale 的表示与导出的权重同格式：值 = m * 2^e，m 归一化到 [2^30, 2^31)。
 * 除法一律「四舍六入五成双」，与训练侧 torch.round / Python round 的中点位取偶一致。
 *
 * 为什么需要 128 位中间量：per-tensor 动态量化的步长是 max(max*128, |min|*127)/16256，
 * 而每个待量化的值本身是 acc * m（|acc| ≤ 2^19、m < 2^31）再对齐到公共指数，实测 |值|
 * 能到 2^51，乘 16256 就是 2^65 —— 超出 int64。这里不依赖 __int128：把乘积拆成高低两半，
 * 再用「减法计数」求商（商 ≤ 128），全部 32/64 位运算，Cortex-M3 上同样能跑。
 */
#ifndef NANOMEOW_FIXED_H
#define NANOMEOW_FIXED_H

#include <stdint.h>

#define NM_MANT_BITS 30
#define NM_MANT_MIN (1 << NM_MANT_BITS)
#define NM_MANT_MAX ((1 << 31) - 1)
#define NM_QMAX 127
#define NM_QUANT_DEN (NM_QMAX * (NM_QMAX + 1))   /* 127 * 128 = 16256 */

typedef struct {
    int32_t m;
    int32_t e;
} nm_scale;

/* 越界哨兵：只有「本不该发生」的钳位会置 1（比如对齐左移超过 int64 能表示的范围）。
 * G1 对拍跑完必须断言它仍是 0 —— 静默钳位就等于静默改行为。 */
extern int nm_range_error;

static inline uint64_t nm_clz64(uint64_t x)
{
    return x == 0 ? 64u : (uint64_t)__builtin_clzll(x);
}

/* 输入：两个 uint64；输出：乘积的高 64 位与低 64 位。预期行为：按 32 位肢拆开算，不依赖 __int128。 */
static inline void nm_mul_wide(uint64_t a, uint64_t b, uint64_t *hi, uint64_t *lo)
{
    uint64_t a0 = a & 0xffffffffULL, a1 = a >> 32;
    uint64_t b0 = b & 0xffffffffULL, b1 = b >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffULL) + (p10 & 0xffffffffULL);
    *lo = (mid << 32) | (p00 & 0xffffffffULL);
    *hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

/* 输入：整数分子 num（可负）、正数分母 den；输出：round(num / den)，四舍六入五成双。
 * 预期行为：与 Python 的 divmod + 取偶一致；den 必须 > 0。 */
static inline int64_t nm_round_div(int64_t num, int64_t den)
{
    if (den > 0 && (den & (den - 1)) == 0) {
        /* den 是 2 的幂：用移位代替 64 位除法（Cortex-M3 上 __aeabi_ldivmod 上百周期）。
         * GCC/Clang 的算术右移对负数就是 floor，补码下 num & (den-1) 正好是 num mod den
         * 的非负余数，所以下面三个式子和通用分支逐位等价。 */
        int k = 63 - (int)__builtin_clzll((uint64_t)den);
        int64_t q = num >> k;
        int64_t r = num & (den - 1);
        int64_t twice = r << 1;
        if (twice > den || (twice == den && (q & 1))) q += 1;
        return q;
    }
    int64_t q = num / den;
    int64_t r = num % den;
    if (r < 0) { r += den; q -= 1; }              /* 转成 floor 语义 */
    int64_t twice = 2 * r;
    if (twice > den || (twice == den && (q & 1))) q += 1;
    return q;
}

/* 输入：128 位正整数 (hi, lo)、指数 e；输出：归一化后的 (m, e)，m ∈ [2^30, 2^31)。
 * 预期行为：与 Python 的 normalize 完全一致 —— 左移补到 2^30、右移（丢低位）压到 2^31 以下，
 *           即「取最高 31 位」；m == 0 时返回 (0, 0)。 */
static inline nm_scale nm_normalize_u128(uint64_t hi, uint64_t lo, int32_t e)
{
    nm_scale out;
    if (hi == 0 && lo == 0) { out.m = 0; out.e = 0; return out; }
    int bitlen = hi ? (int)(128 - nm_clz64(hi)) : (int)(64 - nm_clz64(lo));
    if (bitlen > 31) {
        int k = bitlen - 31;
        if (k >= 64) { lo = hi >> (k - 64); hi = 0; }
        else { lo = (lo >> k) | (k ? (hi << (64 - k)) : 0); hi >>= k; }
        e += k;
    } else if (bitlen < 31) {
        int k = 31 - bitlen;
        hi = (hi << k) | (k ? (lo >> (64 - k)) : 0);
        lo <<= k;
        e -= k;
    }
    out.m = (int32_t)lo;
    out.e = e;
    return out;
}

static inline nm_scale nm_normalize_i64(int64_t m, int32_t e)
{
    uint64_t mag = (uint64_t)(m < 0 ? -m : m);
    nm_scale s = nm_normalize_u128(0, mag, e);
    s.m = (int32_t)(m < 0 ? -(int64_t)s.m : (int64_t)s.m);
    return s;
}

/* 输入：两个 scale；输出：乘积 scale（值 = value(a) * value(b)）。 */
static inline nm_scale nm_mul(nm_scale a, nm_scale b)
{
    uint64_t am = (uint64_t)(a.m < 0 ? -a.m : a.m);
    uint64_t bm = (uint64_t)(b.m < 0 ? -b.m : b.m);
    uint64_t hi, lo;
    nm_mul_wide(am, bm, &hi, &lo);                /* am, bm < 2^31 → hi 恒为 0 */
    nm_scale s = nm_normalize_u128(hi, lo, a.e + b.e);
    s.m = ((a.m < 0) != (b.m < 0)) ? -s.m : s.m;
    return s;
}

/* 输入：两个 scale；输出：商 scale（值 = value(a) / value(b)）。 */
static inline nm_scale nm_div(nm_scale a, nm_scale b)
{
    uint64_t am = (uint64_t)(a.m < 0 ? -a.m : a.m);
    uint64_t bm = (uint64_t)(b.m < 0 ? -b.m : b.m);
    uint64_t num = am << NM_MANT_BITS;            /* am < 2^31 → 左移 30 位后 < 2^61 */
    nm_scale s = nm_normalize_u128(0, num / bm, a.e - b.e - NM_MANT_BITS);
    s.m = ((a.m < 0) != (b.m < 0)) ? -s.m : s.m;
    return s;
}

/* 输入：scale、整数 k；输出：scale * k。 */
static inline nm_scale nm_mul_int(nm_scale a, int64_t k)
{
    uint64_t am = (uint64_t)(a.m < 0 ? -a.m : a.m);
    uint64_t kk = (uint64_t)(k < 0 ? -k : k);
    uint64_t hi, lo;
    nm_mul_wide(am, kk, &hi, &lo);
    nm_scale s = nm_normalize_u128(hi, lo, a.e);
    s.m = ((a.m < 0) != (k < 0)) ? -s.m : s.m;
    return s;
}

/* 输入：scale、正整数 k；输出：scale / k（乘子按四舍六入五成双）。
 * 预期行为：先把乘子左移 32 位再除，否则商的低位会在归一化时被整段丢掉。 */
static inline nm_scale nm_div_int(nm_scale a, int64_t k)
{
    uint64_t am = (uint64_t)(a.m < 0 ? -a.m : a.m);
    uint64_t kk = (uint64_t)(k < 0 ? -k : k);
    uint64_t num = am << 32;                      /* am < 2^31 → < 2^63 */
    int64_t q = nm_round_div((int64_t)num, (int64_t)kk);
    nm_scale s = nm_normalize_u128(0, (uint64_t)q, a.e - 32);
    s.m = ((a.m < 0) != (k < 0)) ? -s.m : s.m;
    return s;
}

/* 输入：已算好的乘积 prod、左移位数 shift；输出：prod << shift。
 * 预期行为：溢出时置越界哨兵并饱和到 INT64 端点 —— 本模型的实际指数域不会触发，
 *           一旦触发说明权重或 scale 出了范围，G1 对拍会以 nm_range_error 报出来。 */
static inline int64_t nm_shl_checked(int64_t prod, int shift)
{
    if (shift <= 0) return prod;
    if (shift >= 62) { nm_range_error = 1; return prod > 0 ? INT64_MAX : INT64_MIN; }
    int64_t limit = INT64_MAX >> shift;
    if (prod > limit || prod < -limit - 1) {
        nm_range_error = 1;
        return prod > 0 ? INT64_MAX : INT64_MIN;
    }
    return prod << shift;
}

/* 输入：无符号 128 位 x（hi:lo）、右移位数 k（>= 1）；输出：round(x / 2^k)，四舍六入五成双。
 * 预期行为：与 Python 的 round_div(x, 1 << k) 一致。x 先拆成商与余数，
 *           再看余数最高位（half）与其余低位（rest）决定是否进位 —— 正好是取偶语义。 */
static inline uint64_t nm_u128_shr_round(uint64_t hi, uint64_t lo, int k)
{
    uint64_t q, half, rest_nonzero;
    if (k >= 128) return 0;
    if (k > 64) {
        int sh = k - 64;                       /* 1..63 */
        q = hi >> sh;
        half = (hi >> (sh - 1)) & 1;
        rest_nonzero = (sh == 1) ? (lo != 0)
                                 : (((hi & (((uint64_t)1 << (sh - 1)) - 1)) != 0) || (lo != 0));
    } else if (k == 64) {
        q = hi;
        half = (lo >> 63) & 1;
        rest_nonzero = (lo << 1) != 0;
    } else {
        q = (hi << (64 - k)) | (lo >> k);
        half = (lo >> (k - 1)) & 1;
        rest_nonzero = (k == 1) ? 0 : ((lo & (((uint64_t)1 << (k - 1)) - 1)) != 0);
    }
    if (half && (rest_nonzero || (q & 1))) q++;
    return q;
}

/* 输入：可负整数 val、scale s；输出：round(val * value(s))。
 *
 * 预期行为：与 Python 的 apply_scale 逐位一致。必须全程 128 位 —— 未量化的中间码
 * 能到 2^59，乘上 m < 2^31 就是 2^90，int64 在乘的那一刻就溢出了。结果装不进 int64
 * 时置越界哨兵并饱和（本模型的实际指数域不会触发）。 */
static inline int64_t nm_apply_scale(int64_t val, nm_scale s)
{
    int neg = (val < 0) != (s.m < 0);
    uint64_t av = (uint64_t)(val < 0 ? -val : val);
    uint64_t am = (uint64_t)(s.m < 0 ? -s.m : s.m);
    uint64_t hi, lo, q;
    int bits;
    if (__builtin_mul_overflow(av, am, &lo)) {
        nm_mul_wide(av, am, &hi, &lo);            /* 只有真正超 64 位才走宽乘法 */
    } else {
        hi = 0;
    }
    if (s.e < 0) {
        q = nm_u128_shr_round(hi, lo, -(int)s.e);
        return neg ? -(int64_t)q : (int64_t)q;
    }
    bits = hi ? (int)(128 - nm_clz64(hi)) : (int)(64 - nm_clz64(lo));
    if (bits + (int)s.e > 63) { nm_range_error = 1; return neg ? INT64_MIN : INT64_MAX; }
    q = s.e > 0 ? (lo << s.e) : lo;      /* bits + e <= 63 已保证不溢出，且此时 hi 必为 0 */
    return neg ? -(int64_t)q : (int64_t)q;
}

/* 输入：整数码 val（实值 = val * value(scale)）、目标小数位 frac_bits；
 * 输出：Q(frac_bits) 整数 = round(val * value(scale) * 2^frac_bits)。 */
static inline int64_t nm_rescale(int64_t val, nm_scale s, int frac_bits)
{
    int64_t shift = (int64_t)s.e + frac_bits;
    int neg = (val < 0) != (s.m < 0);
    uint64_t av = (uint64_t)(val < 0 ? -val : val);
    uint64_t am = (uint64_t)(s.m < 0 ? -s.m : s.m);
    uint64_t hi, lo, q;
    int bits;
    if (__builtin_mul_overflow(av, am, &lo)) {
        nm_mul_wide(av, am, &hi, &lo);            /* 只有真正超 64 位才走宽乘法 */
    } else {
        hi = 0;
    }
    if (shift < 0) {
        q = nm_u128_shr_round(hi, lo, (int)(-shift));
        return neg ? -(int64_t)q : (int64_t)q;
    }
    bits = hi ? (int)(128 - nm_clz64(hi)) : (int)(64 - nm_clz64(lo));
    if (bits + (int)shift > 63) { nm_range_error = 1; return neg ? INT64_MIN : INT64_MAX; }
    q = shift > 0 ? (lo << shift) : lo;  /* bits + shift <= 63 已保证不溢出，且此时 hi 必为 0 */
    return neg ? -(int64_t)q : (int64_t)q;
}

/* 输入：小数位数 frac_bits；输出：值 = 2 ** -frac_bits 的 scale。 */
static inline nm_scale nm_scale_from_frac(int frac_bits)
{
    return nm_normalize_i64(1, -frac_bits);
}

/* 128 位无符号整数：per-tensor 动态量化的步长分子 num = max(max*128, |min|*127)。
 * 未量化的中间量（比如 r * k）的码能到 2^59，乘 128 就是 2^66 —— 超出 int64，
 * 所以这一层必须用 128 位；Python 参考侧是大整数，天然没有这个上限。 */
typedef struct {
    uint64_t hi, lo;
} nm_u128;

/* 输入：两个 128 位无符号数；输出：-1 / 0 / 1（a 小于 / 等于 / 大于 b）。 */
static inline int nm_u128_cmp(nm_u128 a, nm_u128 b)
{
    if (a.hi != b.hi) return a.hi < b.hi ? -1 : 1;
    if (a.lo != b.lo) return a.lo < b.lo ? -1 : 1;
    return 0;
}

/* 输入：128 位无符号数、uint64 乘数；输出：乘积。调用方保证乘积 < 2^128。 */
static inline nm_u128 nm_u128_mul_u64(nm_u128 a, uint64_t m)
{
    uint64_t hi, lo;
    nm_u128 out;
    nm_mul_wide(a.lo, m, &hi, &lo);
    out.lo = lo;
    out.hi = a.hi * m + hi;
    return out;
}

/* 输入：两个 128 位无符号数；输出：a - b。调用方保证 a >= b。 */
static inline nm_u128 nm_u128_sub(nm_u128 a, nm_u128 b)
{
    nm_u128 out;
    out.lo = a.lo - b.lo;
    out.hi = a.hi - b.hi - (a.lo < b.lo ? 1 : 0);
    return out;
}

/* 输入：scale、128 位正整数 k；输出：scale * k（乘子归一化到 [2^30, 2^31)）。 */
static inline nm_scale nm_mul_int_u128(nm_scale a, nm_u128 k)
{
    uint64_t am = (uint64_t)(a.m < 0 ? -a.m : a.m);
    nm_u128 p = nm_u128_mul_u64(k, am);
    nm_scale s = nm_normalize_u128(p.hi, p.lo, a.e);
    s.m = a.m < 0 ? -s.m : s.m;
    return s;
}

/* 输入：可负整数 v、128 位正数 num；输出：round(v * 16256 / num)，取偶。
 *
 * 预期行为：这就是 per-tensor 动态量化的取码式 `round(v * den / num)`（den = 127 * 128）。
 *           商 |q| ≤ 128（因为 num ≥ 127 * max|v|），但中间量 |v| * 16256 能到 2^65，
 *           所以先把 |v| * 16256 拆成 128 位，再用 8 位长除法求商与余数。 */
static inline int64_t nm_requant_code_u128(int64_t v, nm_u128 num)
{
    int neg = v < 0;
    uint64_t av = (uint64_t)(neg ? -v : v);
    uint64_t p_hi = 0, p_lo, q = 0;
    int k, cmp;

    if (av <= (UINT64_MAX / (uint64_t)NM_QUANT_DEN)) {
        p_lo = av * (uint64_t)NM_QUANT_DEN;       /* 单次 64 位乘法，省掉 4 次 32x32 */
    } else {
        nm_mul_wide(av, (uint64_t)NM_QUANT_DEN, &p_hi, &p_lo);
    }

    /* 值域落在 64 位内的快速路径（本模型的 Q16 激活量化点全部走这里）：
     * 同样的 8 位长除法，但每轮只动 64 位，宽运算量减半。条件里的移位前提是
     * num * 128 不溢出 64 位；不满足时自动落到下面的 128 位通用分支。 */
    if (num.hi == 0 && p_hi == 0 && num.lo <= (UINT64_MAX >> 7)) {
        uint64_t d = num.lo, rem = p_lo, t = d << 7;
        for (k = 7; k >= 0; k--) {
            if (rem >= t) { rem -= t; q |= (uint64_t)1 << k; }
            t >>= 1;
        }
        cmp = (rem << 1) > d ? 1 : ((rem << 1) == d ? 0 : -1);
        if (cmp > 0 || (cmp == 0 && (q & 1))) q++;
        return neg ? -(int64_t)q : (int64_t)q;
    }
    {
    nm_u128 p, t, two;
    p.hi = p_hi;
    p.lo = p_lo;
    /* 商 |q| ≤ 128 正好 8 位：从最高位 t = num * 128 起逐位试减（长除法），每轮 t 右移一位。
     * 8 轮定长，取代原来最多 129 次「比较 + 减法」的计数循环。 */
    t = nm_u128_mul_u64(num, 128u);
    for (k = 7; k >= 0; k--) {
        if (nm_u128_cmp(p, t) >= 0) {
            p = nm_u128_sub(p, t);
            q |= (uint64_t)1 << k;
        }
        t.lo = (t.lo >> 1) | (t.hi << 63);
        t.hi >>= 1;
    }
    two.hi = (p.hi << 1) | (p.lo >> 63);          /* 2 * 余数；余数 < num < 2^67 → 不溢出 */
    two.lo = p.lo << 1;
    cmp = nm_u128_cmp(two, num);
    if (cmp > 0 || (cmp == 0 && (q & 1))) q++;
    }
    return neg ? -(int64_t)q : (int64_t)q;
}

/* 输入：可负整数 v、int64 正数 num；输出：round(v * 16256 / num)，取偶。
 * 预期行为：就是 nm_requant_code_u128 在 num 能装进 int64 时的入口。 */
static inline int64_t nm_requant_code(int64_t v, int64_t num)
{
    nm_u128 k;
    k.hi = 0;
    k.lo = (uint64_t)num;
    return nm_requant_code_u128(v, k);
}

/* 输入：两个可负 int64 码；输出：乘积的 128 位大小，符号写进 *neg。
 * 预期行为：值等价于 `a * b`，但不依赖有符号乘法 —— 大小用 nm_mul_wide 精确算，
 *           所以即使乘积到 2^126 也不会踩 UB（本模型实测到 2^63.1）。 */
static inline nm_u128 nm_mul_codes(int64_t a, int64_t b, int *neg)
{
    nm_u128 out;
    uint64_t av = a < 0 ? (uint64_t)0 - (uint64_t)a : (uint64_t)a;
    uint64_t bv = b < 0 ? (uint64_t)0 - (uint64_t)b : (uint64_t)b;
    *neg = (a < 0) != (b < 0);
    nm_mul_wide(av, bv, &out.hi, &out.lo);
    return out;
}

/* 输入：两个 int64 码；输出：a * b，装不进 int64 时置越界哨兵并饱和。
 * 预期行为：正常域内与 `a * b` 同值；溢出说明某个中间张量的码超出设计宽度，
 *           G1 对拍必须看得见，而不是静默回绕。 */
static inline int64_t nm_mul_checked(int64_t a, int64_t b)
{
    int neg;
    nm_u128 p = nm_mul_codes(a, b, &neg);
    if (p.hi != 0 || p.lo > (uint64_t)INT64_MAX + (neg ? 1u : 0u)) {
        nm_range_error = 1;
        return neg ? INT64_MIN : INT64_MAX;
    }
    return neg ? (int64_t)((uint64_t)0 - p.lo) : (int64_t)p.lo;
}

/* 输入：128 位无符号码 k、它的符号、scale s、小数位 frac；
 * 输出：round(±k * value(s) * 2^frac)。
 * 预期行为：与 nm_rescale 同一口径，只是码本身可以是 128 位 —— 两个 QTensor 的码相乘
 *           能到 2^66（Q16 的和 × 未量化的对齐码），落回 int64 之前就得先和 scale 一起算。 */
static inline int64_t nm_rescale_wide(nm_u128 k, int neg, nm_scale s, int frac_bits)
{
    uint64_t am = s.m < 0 ? (uint64_t)0 - (uint64_t)s.m : (uint64_t)s.m;
    nm_u128 p = nm_u128_mul_u64(k, am);
    int shift = s.e + frac_bits;
    uint64_t q;
    int bits;
    if (s.m < 0) neg = !neg;
    if (shift < 0) {
        q = nm_u128_shr_round(p.hi, p.lo, -shift);
        if (q > (uint64_t)INT64_MAX + (neg ? 1u : 0u)) {
            nm_range_error = 1;
            return neg ? INT64_MIN : INT64_MAX;
        }
        return neg ? (int64_t)((uint64_t)0 - q) : (int64_t)q;
    }
    bits = p.hi ? (int)(128 - nm_clz64(p.hi)) : (int)(64 - nm_clz64(p.lo));
    if (bits + shift > 63) { nm_range_error = 1; return neg ? INT64_MIN : INT64_MAX; }
    q = shift > 0 ? (p.lo << shift) : p.lo;
    return neg ? (int64_t)((uint64_t)0 - q) : (int64_t)q;
}

#endif
