/* nanomeow INT8 纯整数推理引擎：与 infer/ref/ 的 Python 定点参考逐位一致（G1 闸门）。
 *
 * 全部运算只有整数：权重 int8、激活 int8 + 一个 scale、wkv state 是 int32
 * （单位 = 写它那一趟 token 的 s_k * s_v）。没有一处浮点。
 *
 * 工作区说明：模型层的张量用文件级 static 暂存（`s_*`），叶子算子的 int64 工作数组也共用
 * 一块文件级暂存，栈上只剩标量。这样栈深度不随层数增长，RAM 占用是一个固定、可量的数
 * （激活 64 位暂存是为了与 Python
 * 的大整数语义逐位对齐；上板按 §10 账本换成 32 位时，这一块的数字要重算）。
 */
#include <limits.h>
#include <string.h>

#include "nanomeow.h"
#include "nm_lut.h"

int nm_range_error = 0;

/* 两张 Q15 表（exp / log1p）放 RAM，由 nm_lut.h 的位流展开而来：原样存要 1,028 B Flash，
 * 现在存每步 2 bit 的二阶差分（138 B），代价是 RAM 多 1,028 B。展开是纯整数、逐项不变。 */
uint16_t nm_exp_lut[NM_LUT_N];
uint16_t nm_log1p_lut[NM_LUT_N];
static int s_lut_ready;

/* 输入：输出缓冲、表长 n、初值 v0、首差 d0、二阶差基值 lo、每步 2 bit 的位流 dd。
 * 输出：无（就地填满 out 的前 n 项）。
 * 预期行为：out[0] = v0；之后 v += d，d 从第 2 步起每步加 (lo + 位流字段)；
 *           与 infer/c/tools/gen_lut.py 的 encode / pack 互逆，解出的表逐项等于 Python 参考。 */
static void nm_lut_expand(uint16_t *out, int n, int v0, int d0, int lo, const uint32_t *dd)
{
    int v = v0, d = d0, i;
    out[0] = (uint16_t)v;
    if (n < 2) return;
    v += d;                          /* 第 1 步用的还是首差 d0，还没吃二阶差字段 */
    out[1] = (uint16_t)v;
    for (i = 2; i < n; i++) {
        int k = i - 2;
        d += lo + (int)((dd[k >> 4] >> ((k & 15) * 2)) & 3u);
        v += d;
        out[i] = (uint16_t)v;
    }
}

/* 输入：无；输出：无。预期行为：把两张表展开进 RAM，只做一次（幂等）；
 * nm_forward_token 每次进来先调它，自检程序也可以直接调来核对表内容。 */
void nm_lut_init(void)
{
    if (s_lut_ready) return;
    nm_lut_expand(nm_exp_lut, NM_LUT_N, nm_exp_lut_v0, nm_exp_lut_d0, nm_exp_lut_lo, nm_exp_lut_dd);
    nm_lut_expand(nm_log1p_lut, NM_LUT_N, nm_log1p_lut_v0, nm_log1p_lut_d0, nm_log1p_lut_lo, nm_log1p_lut_dd);
    s_lut_ready = 1;
}

#define NM_FRAC_BITS 16
#define NM_ONE (1 << NM_FRAC_BITS)
#define NM_LUT_STEP_SHIFT 8
#define NM_LN2_Q16 45426          /* round(ln2 * 2^16) */
#define NM_EPS_Q32 42950          /* round(1e-5 * 2^32)：ln0 / ln1 / ln2 / ln_out */
#define NM_EPS_Q32_X 2748779      /* round(64e-5 * 2^32)：Tmix 的 GroupNorm */
#define NM_MAX_DIM NM_DIM_FFN     /* 最宽的中间量 = CMix 的 key 输出 */

/* 头投影是 256 行，比 nm_tensor 宽，单独放一块暂存（只 nm_head_linear 用） */
static int64_t s_head[NM_VOCAB];

typedef struct { int64_t v[NM_MAX_DIM]; nm_scale scale; } nm_tensor;

/* 叶子算子的共享工作区：这些算子互为叶子（调用图里没有互相嵌套），所以共用一块 ——
 * 原先每个函数都在栈上开 0.5~1.6 KB 的 int64 数组，现在 .bss 里只留一份。
 * 深度 4 是 nm_norm_q 的需求（v / wv / bv / vals）；s_wt / s_wt2 是它和 nm_addmul_param
 * 解包权重用的张量暂存。 */
static int64_t s_wa[NM_MAX_DIM], s_wb[NM_MAX_DIM], s_wc[NM_MAX_DIM], s_wd[NM_MAX_DIM];
static nm_tensor s_wt, s_wt2;
static nm_scale s_sc[NM_MAX_DIM];
static int32_t s_xs[NM_MAX_DIM];

/* ================= 张量与动态量化 ================= */

/* 输入：整数码数组、长度、它的 scale；输出：int8 码 + 新 scale。
 * 预期行为：复刻训练侧 per-tensor 动态假量化（SYMMETRIC_NO_CLIPPING_ERR）——
 *           正端除 127、负端除 128 取大者作步长；全零张量原样返回。 */
static void nm_quantize_dynamic(const int64_t *vals, int n, nm_scale scale,
                                int64_t *out, nm_scale *out_scale)
{
    int i;
    int64_t mx = vals[0], mn = vals[0];
    for (i = 1; i < n; i++) {
        if (vals[i] > mx) mx = vals[i];
        if (vals[i] < mn) mn = vals[i];
    }
    nm_u128 num, other;
    num.hi = 0;
    num.lo = 0;
    if (mx > 0) nm_mul_wide((uint64_t)mx, (uint64_t)(NM_QMAX + 1), &num.hi, &num.lo);
    if (mn < 0) {
        nm_mul_wide((uint64_t)(-mn), (uint64_t)NM_QMAX, &other.hi, &other.lo);
        if (nm_u128_cmp(other, num) > 0) num = other;
    }
    if (num.hi == 0 && num.lo == 0) {
        for (i = 0; i < n; i++) out[i] = 0;
        *out_scale = scale;
        return;
    }
    *out_scale = nm_div_int(nm_mul_int_u128(scale, num), NM_QUANT_DEN);
    if (num.hi == 0 && num.lo == (uint64_t)NM_QUANT_DEN) {
        /* num 正好等于 127*128：说明这组码已经满量程，再量化的码是精确恒等
         * （round(v * 16256 / 16256) == v），只有 scale 要按上面那行重算一次。
         * 本文件里 nm_fq 都是跟在 quantize=1 的算子后面，走的就是这条。 */
        if (out != vals) memcpy(out, vals, (size_t)n * sizeof(int64_t));
        return;
    }
    for (i = 0; i < n; i++) {
        int64_t q = nm_requant_code_u128(vals[i], num);
        out[i] = q < -128 ? -128 : (q > 127 ? 127 : q);
    }
}

/* 输入：张量、长度、目标数组；输出：Q16 整数。预期行为：对应 Python 的 to_fixed。 */
static void nm_to_fixed(const nm_tensor *x, int n, int64_t *out)
{
    int i;
    for (i = 0; i < n; i++) out[i] = nm_rescale(x->v[i], x->scale, NM_FRAC_BITS);
}

/* 输入：Q16 整数、长度、是否量化、输出张量；输出：无。
 * 预期行为：对应 Python 的 _finish —— 量化就走 per-tensor 动态口径（= fq_act），
 *           不量化就保留 scale = 2^-FRAC_BITS。 */
static void nm_finish(const int64_t *vals, int n, int quantize, nm_tensor *out)
{
    if (quantize) {
        nm_quantize_dynamic(vals, n, nm_scale_from_frac(NM_FRAC_BITS), out->v, &out->scale);
    } else {
        memcpy(out->v, vals, (size_t)n * sizeof(int64_t));
        out->scale = nm_scale_from_frac(NM_FRAC_BITS);
    }
}

/* 输入：张量、长度；输出：无。预期行为：就地 per-tensor 动态量化（= fq_act）。
 *           量化只按元素「读旧码、写新码」，num 在循环外就定好，所以能直接原地做 ——
 *           省掉一次 n * 8 字节的整表拷贝（原来先写 tmp 再 memcpy 回来）。 */
static void nm_fq(nm_tensor *x, int n)
{
    nm_quantize_dynamic(x->v, n, x->scale, x->v, &x->scale);
}

/* 输入：张量 a、b；输出：无。预期行为：把 b 原样拷进 a（含 scale）。 */
static void nm_copy(nm_tensor *dst, const nm_tensor *src, int n)
{
    memcpy(dst->v, src->v, (size_t)n * sizeof(int64_t));
    dst->scale = src->scale;
}

/* 输入：字节流、起始位号、位宽 n（1..24）；输出：从该位起、小端位序的 n 位无符号值。
 * 预期行为：只读 ceil((bitpos % 8 + n) / 8) 个字节 —— 位流末尾的那一条不会越界读。
 *           另一种写法是「一次 32 位非对齐读」，交替 A/B 实测只快 1.0%（337.3K vs
 *           340.7K cycles/token），却要多留 3 B 填充，所以这里不用它。 */
static uint32_t nm_read_bits(const uint8_t *p, uint32_t bitpos, uint32_t n)
{
    uint32_t off = bitpos & 7u, nb = (n + off + 7u) >> 3, acc = 0, i;
    const uint8_t *q = p + (bitpos >> 3);
    for (i = 0; i < nb; i++) acc |= (uint32_t)q[i] << (8u * i);
    return (acc >> off) & ((1u << n) - 1u);
}

/* 输入：per-row 的 scale 位流、行数；输出：行索引段相对位流开头的字节偏移。
 * 预期行为：布局 = [ceil(rows*23/8) 尾数][1 B 基线][1 B 增量位宽 w][ceil(rows*w/8) 增量][索引]，
 *           段偏移全部由 rows 与 w 现算，所以描述符里不必再存一个偏移字段。 */
static uint32_t nm_index_off(const uint8_t *p, uint32_t rows)
{
    uint32_t mb = (rows * 23u + 7u) >> 3;
    uint32_t w = p[mb + 1];
    return mb + 2u + (w ? ((rows * w + 7u) >> 3) : 0u);
}

/* 输入：权重矩阵、行号；输出：该行的 scale。per-row 取第 row 行、per-tensor 恒取第 0 个。
 * 预期行为：把位流还原成 (mul, shift)。mul = (2^23 + 尾数 23 位) << 7 —— 归一化后最高位
 *           恒 1，不必存；shift = 每张量的基线 + 行内小位宽增量（位宽 0 表示整张量同一个值）。 */
static nm_scale nm_row_scale(const nm_mat *m, int row)
{
    nm_scale s;
    if (!m->per_row) {
        const uint8_t *q = nm_pool + m->scale_off;
        uint32_t v = (uint32_t)q[0] | ((uint32_t)q[1] << 8)
                     | ((uint32_t)q[2] << 16) | ((uint32_t)q[3] << 24);
        s.m = (int32_t)(((v >> 7) | (1u << 23)) << 7);
        s.e = (int8_t)((v & 0x7Fu) - 127u);
        return s;
    }
    {
        const uint8_t *p = nm_pool + m->scale_off;
        uint32_t rows = (uint32_t)m->rows, mb = (rows * 23u + 7u) >> 3;
        uint32_t w = p[mb + 1];
        uint32_t sh = (uint32_t)(int32_t)(int8_t)p[mb];
        if (w) sh += nm_read_bits(p + mb + 2u, (uint32_t)row * w, w);
        s.m = (int32_t)((nm_read_bits(p, (uint32_t)row * 23u, 23u) | (1u << 23)) << 7);
        s.e = (int8_t)sh;
    }
    return s;
}

/* 输入：权重矩阵、行号；输出：该行 int8 码的首地址。
 * 预期行为：idx_bits = 0 时第 row 行就是连续排布的第 row 行；否则 codes 是去重后的唯一行池，
 *           行号先经位流末尾的索引表映射 —— 省 Flash 的代价是每次取行多一次位流解码。 */
static const int8_t *nm_row_codes(const nm_mat *m, int row)
{
    const uint8_t *p = nm_pool + m->scale_off;
    uint32_t idx = (uint32_t)row;
    if (m->idx_bits)
        idx = nm_read_bits(p + nm_index_off(p, (uint32_t)m->rows),
                           (uint32_t)row * m->idx_bits, m->idx_bits);
    return (const int8_t *)(nm_pool + m->code_off) + (size_t)idx * (uint32_t)m->cols;
}

/* 输入：per-tensor 权重（如 x_r / k_k / r_k）；输出：装成张量的它。 */
static void nm_param(const nm_mat *m, int n, nm_tensor *out)
{
    int i;
    for (i = 0; i < n; i++) out->v[i] = (int8_t)nm_pool[m->code_off + i];
    out->scale = nm_row_scale(m, 0);
}

/* ================= 定点非线性（exp / log1p 两张表） ================= */

/* 输入：Q15 表、表长、Q(step_shift) 的查表位置、步长指数；输出：Q16 的线性插值。
 * 预期行为：与 Python 的 _lerp_q15 一致 —— 位置先钳到右端点之内，低位是插值权重。 */
static int32_t nm_lerp_q15(const uint16_t *lut, int n, int64_t pos, int step_shift)
{
    int64_t max_pos = ((int64_t)(n - 1) << step_shift) - 1;
    int idx;
    int64_t frac;
    int32_t lo, hi;
    if (pos > max_pos) pos = max_pos;
    idx = (int)(pos >> step_shift);
    frac = pos & (((int64_t)1 << step_shift) - 1);
    lo = (int32_t)lut[idx];
    hi = (int32_t)lut[idx + 1];
    return (int32_t)(((int64_t)lo << (NM_FRAC_BITS - NM_EXP_LUT_Q))
                     + nm_round_div((int64_t)(hi - lo) * frac,
                                    1 << (NM_EXP_LUT_Q + step_shift - NM_FRAC_BITS)));
}

/* 输入：Q16 整数 x <= 0；输出：floor(x / ln2)（非正）。
 * 预期行为：写成「非负除法再取负」，不依赖负数除法的取整方向。 */
static int32_t nm_floor_div_ln2(int32_t x)
{
    return -(int32_t)(((-(int64_t)x) + NM_LN2_Q16 - 1) / NM_LN2_Q16);
}

/* 输入：Q16 整数 x（调用方保证 x <= 0）；输出：Q16 的 exp(x) ∈ (0, 1]。 */
static int32_t nm_exp_q(int32_t x)
{
    int32_t n, r, value;
    int64_t pos;
    if (x >= 0) return NM_ONE;
    n = nm_floor_div_ln2(x);
    if (n < -NM_FRAC_BITS - 1) return 0;
    r = x - n * NM_LN2_Q16;
    pos = nm_round_div((int64_t)r << 16, NM_LN2_Q16);
    value = NM_ONE + nm_lerp_q15(nm_exp_lut, 257, pos, NM_LUT_STEP_SHIFT);
    if (n == 0) return value;
    return (int32_t)nm_round_div(value, (int64_t)1 << (-n));
}

/* 输入：Q16 整数 t ∈ [0, 1]；输出：Q16 的 log(1 + t) ∈ [0, ln2]。 */
static int32_t nm_log1p_q(int32_t t)
{
    if (t <= 0) return 0;
    if (t > NM_ONE) t = NM_ONE;
    return nm_lerp_q15(nm_log1p_lut, 257, t, NM_LUT_STEP_SHIFT);
}

/* 输入：Q16 整数；输出：Q16 的 sigmoid(x) ∈ (0, 1)。 */
static int32_t nm_sigmoid_q(int32_t x)
{
    int32_t e;
    if (x >= 0) return (int32_t)nm_round_div((int64_t)NM_ONE << NM_FRAC_BITS,
                                             NM_ONE + nm_exp_q(-x));
    e = nm_exp_q(x);
    return (int32_t)nm_round_div((int64_t)e << NM_FRAC_BITS, NM_ONE + e);
}

/* 输入：Q16 整数；输出：Q16 的 tanh(x) ∈ (-1, 1)。预期行为：|x| > 16 先夹住（早就饱和了）。 */
static int32_t nm_tanh_q(int32_t x)
{
    int32_t limit = 16 * NM_ONE;
    if (x > limit) x = limit;
    else if (x < -limit) x = -limit;
    return 2 * nm_sigmoid_q(2 * x) - NM_ONE;
}

/* 输入：Q16 整数；输出：Q16 的 softplus(x) >= 0。预期行为：两侧 16 之外饱和。 */
static int32_t nm_softplus_q(int32_t x)
{
    int32_t limit = 16 * NM_ONE;
    if (x >= limit) return x;
    if (x <= -limit) return 0;
    if (x <= 0) return nm_log1p_q(nm_exp_q(x));
    return x + nm_log1p_q(nm_exp_q(-x));
}

/* ================= 归一化 ================= */

/* 输入：非负整数 n；输出：floor(sqrt(n))。预期行为：牛顿法，初值取 2^ceil(bitlen/2)（不小于真实根）。 */
static uint32_t nm_isqrt(uint64_t n)
{
    uint64_t x, y;
    if (n == 0) return 0;
    x = (uint64_t)1 << (((64 - nm_clz64(n)) + 1) / 2);
    for (;;) {
        y = (x + n / x) >> 1;
        if (y >= x) return (uint32_t)x;
        x = y;
    }
}

/* 输入：Q32 的正整数 x；输出：Q16 的 1 / sqrt(x / 2^32)。
 * 预期行为：先把被开方数左移 2*extra 位再开方，最后带舍入地除一次。 */
static int32_t nm_rsqrt_q32(int64_t x)
{
    const int extra = 10;
    uint64_t s;
    if (x <= 0) return 0;
    s = nm_isqrt((uint64_t)x << (2 * extra));
    return (int32_t)nm_round_div((int64_t)1 << (2 * NM_FRAC_BITS + extra), (int64_t)s);
}

/* 输入：张量 x、per-tensor 的 weight / bias、eps 的 Q32、组数、长度；输出：张量。
 * 预期行为：逐组 (x - mean) * rsqrt(var + eps) * w + b，最后 per-tensor 动态量化；
 *           与 Python 的 norm_q 一致（LayerNorm 就是 groups = 1，方差用有偏口径）。 */
static void nm_norm_q(const nm_tensor *x, const nm_mat *w, const nm_mat *b,
                      int32_t eps_q32, int groups, int n, nm_tensor *out)
{
    int64_t *v = s_wa, *wv = s_wb, *bv = s_wc, *vals = s_wd;
    int size = n / groups, gi, i, k;
    nm_to_fixed(x, n, v);
    nm_param(w, n, &s_wt);
    nm_param(b, n, &s_wt2);
    nm_to_fixed(&s_wt, n, wv);
    nm_to_fixed(&s_wt2, n, bv);
    for (gi = 0; gi < groups; gi++) {
        int base = gi * size;
        int64_t sum = 0, acc = 0, mean, var_q32;
        int32_t r;
        for (i = 0; i < size; i++) sum += v[base + i];
        mean = nm_round_div(sum, size);
        for (i = 0; i < size; i++) {
            int64_t d = v[base + i] - mean;
            acc += d * d;
        }
        var_q32 = nm_round_div(acc, size);
        r = nm_rsqrt_q32(var_q32 + eps_q32);
        for (i = 0; i < size; i++) {
            k = base + i;
            vals[k] = nm_round_div(nm_round_div((v[k] - mean) * r, NM_ONE) * wv[k], NM_ONE) + bv[k];
        }
    }
    nm_finish(vals, n, 1, out);
}

/* ================= 逐元素算子 ================= */

static void nm_neg_t(const nm_tensor *a, int n, nm_tensor *out)
{
    int i;
    for (i = 0; i < n; i++) out->v[i] = -a->v[i];
    out->scale = a->scale;
}

static void nm_add_t(const nm_tensor *a, const nm_tensor *b, int n, int quantize, nm_tensor *out)
{
    int64_t *va = s_wa, *vb = s_wb, *vals = s_wc;
    int i;
    nm_to_fixed(a, n, va);
    nm_to_fixed(b, n, vb);
    for (i = 0; i < n; i++) vals[i] = va[i] + vb[i];
    nm_finish(vals, n, quantize, out);
}

static void nm_sub_t(const nm_tensor *a, const nm_tensor *b, int n, int quantize, nm_tensor *out)
{
    int64_t *va = s_wa, *vb = s_wb, *vals = s_wc;
    int i;
    nm_to_fixed(a, n, va);
    nm_to_fixed(b, n, vb);
    for (i = 0; i < n; i++) vals[i] = va[i] - vb[i];
    nm_finish(vals, n, quantize, out);
}

/* 输入：张量、有理数 num/den；输出：a + num/den。预期行为：对应 `-softplus(...) - 0.5` 这类常量加。 */
static void nm_add_real(const nm_tensor *a, int n, int64_t num, int64_t den, int quantize,
                        nm_tensor *out)
{
    int64_t *va = s_wa, *vals = s_wb;
    int64_t c = nm_round_div(num * NM_ONE, den);
    int i;
    nm_to_fixed(a, n, va);
    for (i = 0; i < n; i++) vals[i] = va[i] + c;
    nm_finish(vals, n, quantize, out);
}

/* 输入：三个张量；输出：a + b * c。预期行为：对应 time-shift 混合与 value residual 的 lerp。 */
static void nm_addmul(const nm_tensor *a, const nm_tensor *b, const nm_tensor *c, int n,
                      int quantize, nm_tensor *out)
{
    int64_t *va = s_wa, *vb = s_wb, *vc = s_wc, *vals = s_wd;
    int i;
    nm_to_fixed(a, n, va);
    nm_to_fixed(b, n, vb);
    nm_to_fixed(c, n, vc);
    for (i = 0; i < n; i++) vals[i] = va[i] + nm_round_div(vb[i] * vc[i], NM_ONE);
    nm_finish(vals, n, quantize, out);
}

/* 输入：张量；输出：relu(x)^2。预期行为：对应 CMix 的 `fq_act(relu(key(k)) ** 2)`。 */
static void nm_relu_sq(const nm_tensor *x, int n, int quantize, nm_tensor *out)
{
    int64_t *v = s_wa, *vals = s_wb;
    int i;
    nm_to_fixed(x, n, v);
    for (i = 0; i < n; i++) {
        int64_t t = v[i] < 0 ? 0 : v[i];
        vals[i] = nm_round_div(t * t, NM_ONE);
    }
    nm_finish(vals, n, quantize, out);
}

/* 输入：张量（长度 = head 数 * 每个 head 的长度）；输出：每个 head 内求和后广播回该 head。 */
static void nm_sum_head(const nm_tensor *x, int head_size, int n, int quantize, nm_tensor *out)
{
    int64_t *v = s_wa, *vals = s_wb;
    int base, i;
    nm_to_fixed(x, n, v);
    for (base = 0; base < n; base += head_size) {
        int64_t s = 0;
        for (i = 0; i < head_size; i++) s += v[base + i];
        for (i = 0; i < head_size; i++) vals[base + i] = s;
    }
    nm_finish(vals, n, quantize, out);
}

/* 输入：两个张量；输出：逐元素相乘（scale 相乘），可选再量化。 */
static void nm_mul_t(const nm_tensor *a, const nm_tensor *b, int n, int quantize, nm_tensor *out)
{
    int i;
    for (i = 0; i < n; i++) out->v[i] = nm_mul_checked(a->v[i], b->v[i]);
    out->scale = nm_mul(a->scale, b->scale);
    if (quantize) nm_fq(out, n);
}

/* 输入：Q(FRAC_BITS) 张量 a、两个张量 b / c、长度、是否量化；输出：fq(a + b * c)。
 * 预期行为：对应 Python 的 `fq(qt_add(a, qt_mul(b, c, quantize=False)))`。
 *           b 与 c 的码相乘能到 2^66（`sum_head(rk)` 是 Q16 码、`v` 是未量化的对齐码），
 *           装不进 int64，所以这里不落中间张量，直接用 128 位算 b * c 的 Q(FRAC_BITS) 值。 */
static void nm_add_mulq(const nm_tensor *a, const nm_tensor *b, const nm_tensor *c, int n,
                        int quantize, nm_tensor *out)
{
    int64_t *va = s_wa, *vals = s_wb;
    nm_scale s = nm_mul(b->scale, c->scale);
    int i;
    nm_to_fixed(a, n, va);
    for (i = 0; i < n; i++) {
        int neg;
        nm_u128 k = nm_mul_codes(b->v[i], c->v[i], &neg);
        vals[i] = va[i] + nm_rescale_wide(k, neg, s, NM_FRAC_BITS);
    }
    nm_finish(vals, n, quantize, out);
}

/* 输入：张量 a、b、per-tensor 权重 w；输出：fq(a + b * w)。对应 time-shift 的 x_r / x_k 那几路。 */
static void nm_addmul_param(const nm_tensor *a, const nm_tensor *b, const nm_mat *w, int n,
                            nm_tensor *out)
{
    nm_param(w, n, &s_wt);
    nm_addmul(a, b, &s_wt, n, 1, out);
    nm_fq(out, n);
}

static void nm_sigmoid_t(const nm_tensor *x, int n, int quantize, nm_tensor *out)
{
    int64_t *v = s_wa, *vals = s_wb;
    int i;
    nm_to_fixed(x, n, v);
    for (i = 0; i < n; i++) vals[i] = nm_sigmoid_q((int32_t)v[i]);
    nm_finish(vals, n, quantize, out);
}

static void nm_tanh_t(const nm_tensor *x, int n, int quantize, nm_tensor *out)
{
    int64_t *v = s_wa, *vals = s_wb;
    int i;
    nm_to_fixed(x, n, v);
    for (i = 0; i < n; i++) vals[i] = nm_tanh_q((int32_t)v[i]);
    nm_finish(vals, n, quantize, out);
}

static void nm_softplus_t(const nm_tensor *x, int n, int quantize, nm_tensor *out)
{
    int64_t *v = s_wa, *vals = s_wb;
    int i;
    nm_to_fixed(x, n, v);
    for (i = 0; i < n; i++) vals[i] = nm_softplus_q((int32_t)v[i]);
    nm_finish(vals, n, quantize, out);
}

/* ================= 整数 GEMV + 动态再量化 ================= */

/* 输入：int8 码行（成对版本给两行）、int32 输入、列数；输出：点积（成对版本写进出参）。
 * 预期行为：与逐项 `a += row[j] * xs[j]` 完全同序同值 —— 每一行仍然是从 j = 0 加到 n-1 的
 *           精确 int32 累加，只是换了循环结构，所以结果逐位相同。
 * 两行版本让同一列的两个 x 只载入一次：Cortex-M3 上实测从 5 条/MAC 降到 3.5 条。
 * 循环写成 do-while 而不是 for：顶部判断 + 底部回跳会多一条分支（实测 6 条/MAC）。
 * 用 always_inline 是因为 -Oz 会把普通 static inline 变成逐行函数调用，反而更慢。 */
static inline __attribute__((always_inline))
int32_t nm_dot_i8(const int8_t *row, const int32_t *xs, int n)
{
    const int8_t *rp = row;
    const int32_t *xp = xs;
    int32_t a = 0;
    int j = n;

    if (j > 0) {
        do {
            a += (int32_t)*rp++ * *xp++;
        } while (--j);
    }
    return a;
}

static inline __attribute__((always_inline))
void nm_dot2_i8(const int8_t *ra, const int8_t *rb, const int32_t *xs, int n,
                int32_t *out_a, int32_t *out_b)
{
    const int32_t *xp = xs;
    int32_t a = 0, b = 0;
    int j = n;

    if (j > 0) {
        do {
            int32_t x = *xp++;
            a += (int32_t)*ra++ * x;
            b += (int32_t)*rb++ * x;
        } while (--j);
    }
    *out_a = a;
    *out_b = b;
}

/* 输入：输入张量、per-row 权重、是否量化输出；输出：张量。
 * 预期行为：acc = Σ w * x；每行实值 = acc * s_w * s_x，先把各行对齐到公共指数（= Python 的
 *           MAX_ALIGN_SHIFT 口径），quantize=True 时再按 per-tensor 动态口径量化回 int8。 */
static void nm_linear(const nm_tensor *x, const nm_mat *m, int quantize, nm_tensor *out)
{
    int64_t *acc = s_wa, *vals = s_wb;
    nm_scale *sc = s_sc;
    int32_t *xs = s_xs;
    int32_t e_ref = INT32_MAX;
    int r, j;
    /* 输入码先落成 int32：内层是 int8 x int8 的点积，用 int64 数组会退化成 64 位乘加
     * （Cortex-M3 上是 __aeabi_lmul 软件例程）。 */
    for (j = 0; j < m->cols; j++) xs[j] = (int32_t)x->v[j];
    for (r = 0; r < m->rows; r++) {
        sc[r] = nm_mul(nm_row_scale(m, r), x->scale);
        if (sc[r].e < e_ref) e_ref = sc[r].e;
    }
    for (r = 0; r + 1 < m->rows; r += 2) {
        int32_t a, b;
        nm_dot2_i8(nm_row_codes(m, r), nm_row_codes(m, r + 1), xs, m->cols, &a, &b);
        acc[r] = a;
        acc[r + 1] = b;
    }
    if (r < m->rows)
        acc[r] = nm_dot_i8(nm_row_codes(m, r), xs, m->cols);
    for (r = 0; r < m->rows; r++) {
        int sh = sc[r].e - e_ref;
        if (sh > 40) { nm_range_error = 1; sh = 40; }
        vals[r] = nm_shl_checked(acc[r] * sc[r].m, sh);
    }
    if (!quantize) {
        memcpy(out->v, vals, (size_t)m->rows * sizeof(int64_t));
        out->scale = nm_normalize_i64(1, e_ref);
        return;
    }
    nm_quantize_dynamic(vals, m->rows, nm_normalize_i64(1, e_ref), out->v, &out->scale);
}

/* 输入：输入张量、per-row 的 head 权重、输出码数组与 scale；输出：无。
 * 预期行为：与 nm_linear(..., quantize=1) 同一口径，只是行数是 NM_VOCAB（256），
 *           比 nm_tensor 的宽度大，所以用一块专门的暂存，结果直接落成 int8 码。
 *           公共指数先扫一遍 scale（与行累加值无关），再算每行、取步长、取码。 */
static void nm_head_linear(const nm_tensor *x, const nm_mat *m, int8_t *codes, nm_scale *scale)
{
    int32_t e_ref = INT32_MAX;
    int64_t mx, mn;
    nm_u128 num, other;
    nm_scale base;
    int32_t *xs = s_xs;
    int r, j;
    for (j = 0; j < m->cols; j++) xs[j] = (int32_t)x->v[j];
    for (r = 0; r < m->rows; r++) {
        nm_scale sc = nm_mul(nm_row_scale(m, r), x->scale);
        if (sc.e < e_ref) e_ref = sc.e;
    }
    for (r = 0; r < m->rows; r++) {
        int32_t a, sh;
        nm_scale sc = nm_mul(nm_row_scale(m, r), x->scale);
        a = nm_dot_i8(nm_row_codes(m, r), xs, m->cols);
        sh = sc.e - e_ref;
        if (sh > 40) { nm_range_error = 1; sh = 40; }
        s_head[r] = nm_shl_checked((int64_t)a * sc.m, sh);
    }
    mx = s_head[0];
    mn = s_head[0];
    for (r = 1; r < m->rows; r++) {
        if (s_head[r] > mx) mx = s_head[r];
        if (s_head[r] < mn) mn = s_head[r];
    }
    num.hi = 0;
    num.lo = 0;
    if (mx > 0) nm_mul_wide((uint64_t)mx, (uint64_t)(NM_QMAX + 1), &num.hi, &num.lo);
    if (mn < 0) {
        nm_mul_wide((uint64_t)(-mn), (uint64_t)NM_QMAX, &other.hi, &other.lo);
        if (nm_u128_cmp(other, num) > 0) num = other;
    }
    base = nm_normalize_i64(1, e_ref);
    if (num.hi == 0 && num.lo == 0) {
        for (r = 0; r < m->rows; r++) codes[r] = 0;
        *scale = base;
        return;
    }
    *scale = nm_div_int(nm_mul_int_u128(base, num), NM_QUANT_DEN);
    for (r = 0; r < m->rows; r++) {
        int64_t q = nm_requant_code_u128(s_head[r], num);
        codes[r] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }
}

/* ================= wkv7 ================= */

/* 输入：张量（长度 = groups * 组大小）、组数、长度；输出：张量。
 * 预期行为：复刻 `fq_act(F.normalize(x.view(..., groups, -1), dim=-1, p=2))`；
 *           范数为 0 的组输出全 0。 */
static void nm_normalize_p2(const nm_tensor *x, int groups, int n, nm_tensor *out)
{
    int64_t *v = s_wa, *vals = s_wb;
    int size = n / groups, gi, i;
    nm_to_fixed(x, n, v);
    for (gi = 0; gi < groups; gi++) {
        int base = gi * size;
        int64_t sos = 0;
        uint32_t norm_q16;
        for (i = 0; i < size; i++) sos += v[base + i] * v[base + i];
        norm_q16 = nm_isqrt((uint64_t)sos);
        if (norm_q16 == 0) {
            for (i = 0; i < size; i++) vals[base + i] = 0;
            continue;
        }
        for (i = 0; i < size; i++)
            vals[base + i] = nm_round_div(v[base + i] * NM_ONE, (int64_t)norm_q16);
    }
    nm_finish(vals, n, 1, out);
}

/* 输入：w_in 的张量（恒 <= -0.5）、长度；输出：Q15 衰减码 = round(exp(-exp(w_in)) * 2^15)。 */
static void nm_decay_q15(const nm_tensor *w, int n, int32_t *w15)
{
    int64_t *v = s_wa;
    int i;
    nm_to_fixed(w, n, v);
    for (i = 0; i < n; i++) {
        int32_t log_w = -nm_exp_q((int32_t)v[i]);
        w15[i] = (int32_t)nm_round_div(nm_exp_q(log_w), 2);
    }
}

/* 输入：q/k/v/a/b 张量、Q15 衰减、int32 state（就地更新）、head 边长、长度、
 *       上一趟的 state 单位与有效性；输出：y 张量（scale = s_q * s_k * s_v）。
 * 预期行为：逐 token 逐 head 跑「sa → 更新 state → y」，全部整数；旧 state 先从 step_prev
 *           换算到当前 step（训练侧每个 chunk 边界都会按当前 state_step 重新吸附）。 */
static void nm_wkv7_recurrence(const nm_tensor *q, const nm_tensor *k, const nm_tensor *v,
                               const nm_tensor *a, const nm_tensor *b, const int32_t *w15,
                               int32_t state[NM_N_HEAD][NM_HEAD_SIZE][NM_HEAD_SIZE],
                               int head_size, int n, const nm_scale *step_prev,
                               int step_prev_valid, nm_tensor *out)
{
    int n_head = n / head_size;
    nm_scale step = nm_mul(k->scale, v->scale);
    nm_scale ab_scale = nm_mul(b->scale, a->scale);
    int64_t *ys = s_wa;
    int h, i, j;
    if (step_prev_valid) {
        nm_scale ratio = nm_div(*step_prev, step);
        for (h = 0; h < n_head; h++)
            for (i = 0; i < head_size; i++)
                for (j = 0; j < head_size; j++)
                    state[h][i][j] = (int32_t)nm_apply_scale(state[h][i][j], ratio);
    }
    for (h = 0; h < n_head; h++) {
        int base = h * head_size;
        int64_t sa[NM_HEAD_SIZE];
        for (j = 0; j < head_size; j++) sa[j] = 0;
        for (i = 0; i < head_size; i++) {
            int64_t ai = a->v[base + i];
            if (ai)
                for (j = 0; j < head_size; j++) sa[j] += ai * state[h][i][j];
        }
        for (i = 0; i < head_size; i++) {
            int64_t w_i = w15[base + i], b_i = b->v[base + i], k_i = k->v[base + i];
            for (j = 0; j < head_size; j++) {
                int64_t val = nm_round_div((int64_t)state[h][i][j] * w_i, 1 << 15);
                if (b_i && sa[j]) val += nm_apply_scale(b_i * sa[j], ab_scale);
                if (k_i) val += k_i * v->v[base + j];
                state[h][i][j] = (int32_t)val;
            }
        }
        for (j = 0; j < head_size; j++) {
            int64_t acc = 0;
            for (i = 0; i < head_size; i++) acc += q->v[base + i] * state[h][i][j];
            ys[base + j] = acc;
        }
    }
    memcpy(out->v, ys, (size_t)n * sizeof(int64_t));
    out->scale = nm_mul(q->scale, step);
}

/* ================= 模型 ================= */

static nm_tensor s_xx, s_xr, s_xw, s_xk, s_xv, s_xa, s_xg;
static nm_tensor s_r, s_q, s_decay, s_w, s_k, s_v, s_vq, s_gate, s_a, s_g;
static nm_tensor s_kk, s_kq, s_y, s_rk, s_tmp, s_tmp2, s_k64;
static int32_t s_w15[NM_N_EMBD];

/* 输入：token；输出：该行权重的张量（per-row 量化后查表，查表本身不做激活量化）。 */
static void nm_embed(int token, nm_tensor *out)
{
    int j;
    for (j = 0; j < NM_N_EMBD; j++)
        out->v[j] = (int8_t)nm_pool[nm_emb_weight.code_off + token * NM_N_EMBD + j];
    out->scale = nm_row_scale(&nm_emb_weight, token);
}

/* 输入：层号、ln1 输出、v_first（可为空）、该层状态；输出：att 输出（v_first 就地更新）。
 * 预期行为：逐行复刻 infer/ref/model.py 的 Int8Model._att，量化点一一对应。 */
static void nm_att(int layer, const nm_tensor *h1, nm_tensor *v_first, int *v_first_valid,
                   nm_layer_state *st, nm_tensor *out)
{
    const nm_block *p = &nm_blocks[layer];
    static nm_tensor prev;

    /* xx = fq(prev - x)，首 token 时 prev 视作 0 */
    if (st->att_prev_valid) {
        for (int j = 0; j < NM_N_EMBD; j++) prev.v[j] = st->att_prev[j];
        prev.scale = st->att_prev_scale;
        nm_sub_t(&prev, h1, NM_N_EMBD, 1, &s_xx);
    } else {
        nm_neg_t(h1, NM_N_EMBD, &s_xx);
    }
    nm_fq(&s_xx, NM_N_EMBD);

    nm_addmul_param(h1, &s_xx, &p->x_r, NM_N_EMBD, &s_xr);
    nm_addmul_param(h1, &s_xx, &p->x_w, NM_N_EMBD, &s_xw);
    nm_addmul_param(h1, &s_xx, &p->x_k, NM_N_EMBD, &s_xk);
    nm_addmul_param(h1, &s_xx, &p->x_v, NM_N_EMBD, &s_xv);
    nm_addmul_param(h1, &s_xx, &p->x_a, NM_N_EMBD, &s_xa);
    nm_addmul_param(h1, &s_xx, &p->x_g, NM_N_EMBD, &s_xg);

    nm_linear(&s_xr, &p->receptance, 0, &s_r);

    /* decay_in = w0 + w2 @ tanh(w1 @ xw)；w = fq(-softplus(-decay_in) - 0.5)。
     * 低秩对的隐藏宽度取权重自己的行数（w1 是 32 -> hid），不能拿 NM_N_EMBD 当长度。 */
    nm_linear(&s_xw, &p->w1, 1, &s_tmp);
    nm_tanh_t(&s_tmp, p->w1.rows, 1, &s_tmp);
    nm_linear(&s_tmp, &p->w2, 1, &s_tmp2);
    nm_param(&p->w0, NM_N_EMBD, &s_tmp);
    nm_add_t(&s_tmp, &s_tmp2, NM_N_EMBD, 0, &s_decay);
    nm_neg_t(&s_decay, NM_N_EMBD, &s_tmp);
    nm_softplus_t(&s_tmp, NM_N_EMBD, 0, &s_tmp);
    nm_neg_t(&s_tmp, NM_N_EMBD, &s_tmp);
    nm_add_real(&s_tmp, NM_N_EMBD, -1, 2, 1, &s_w);
    nm_fq(&s_w, NM_N_EMBD);

    nm_linear(&s_xk, &p->key, 0, &s_k);
    nm_linear(&s_xv, &p->value, 0, &s_v);

    if (layer == 0) {
        nm_copy(v_first, &s_v, NM_N_EMBD);
        *v_first_valid = 1;
    } else {
        nm_linear(&s_xv, &p->v1, 1, &s_tmp);
        nm_linear(&s_tmp, &p->v2, 1, &s_tmp2);
        nm_param(&p->v0, NM_N_EMBD, &s_tmp);
        nm_add_t(&s_tmp, &s_tmp2, NM_N_EMBD, 0, &s_tmp);
        nm_sigmoid_t(&s_tmp, NM_N_EMBD, 1, &s_gate);
        nm_fq(&s_gate, NM_N_EMBD);
        nm_sub_t(v_first, &s_v, NM_N_EMBD, 0, &s_tmp);
        nm_addmul(&s_v, &s_tmp, &s_gate, NM_N_EMBD, 1, &s_tmp);
        nm_fq(&s_tmp, NM_N_EMBD);
        nm_copy(&s_v, &s_tmp, NM_N_EMBD);
    }

    nm_linear(&s_xa, &p->a1, 1, &s_tmp);
    nm_linear(&s_tmp, &p->a2, 1, &s_tmp2);
    nm_param(&p->a0, NM_N_EMBD, &s_tmp);
    nm_add_t(&s_tmp, &s_tmp2, NM_N_EMBD, 0, &s_tmp);
    nm_sigmoid_t(&s_tmp, NM_N_EMBD, 1, &s_a);
    nm_fq(&s_a, NM_N_EMBD);

    nm_linear(&s_xg, &p->g1, 1, &s_tmp);
    nm_sigmoid_t(&s_tmp, p->g1.rows, 1, &s_tmp);
    nm_fq(&s_tmp, p->g1.rows);
    nm_linear(&s_tmp, &p->g2, 1, &s_g);

    /* kk = normalize_p2(fq(k * k_k))；k = fq(k + k * ((a - 1) * k_a)) */
    nm_param(&p->k_k, NM_N_EMBD, &s_tmp2);
    nm_mul_t(&s_k, &s_tmp2, NM_N_EMBD, 1, &s_tmp);
    nm_fq(&s_tmp, NM_N_EMBD);
    nm_normalize_p2(&s_tmp, NM_N_HEAD, NM_N_EMBD, &s_kk);

    nm_copy(&s_kq, &s_k, NM_N_EMBD);
    nm_fq(&s_kq, NM_N_EMBD);
    nm_add_real(&s_a, NM_N_EMBD, -1, 1, 0, &s_tmp);
    nm_param(&p->k_a, NM_N_EMBD, &s_tmp2);
    nm_mul_t(&s_tmp, &s_tmp2, NM_N_EMBD, 0, &s_tmp);
    nm_addmul(&s_kq, &s_kq, &s_tmp, NM_N_EMBD, 1, &s_tmp2);
    nm_fq(&s_tmp2, NM_N_EMBD);
    nm_copy(&s_k, &s_tmp2, NM_N_EMBD);

    /* 递推入口：q = fq(r)、v = fq(v)；残差项用的是**没有再量化**的 r 与 v */
    nm_copy(&s_q, &s_r, NM_N_EMBD);
    nm_fq(&s_q, NM_N_EMBD);
    nm_copy(&s_vq, &s_v, NM_N_EMBD);
    nm_fq(&s_vq, NM_N_EMBD);
    nm_neg_t(&s_kk, NM_N_EMBD, &s_tmp);
    nm_fq(&s_tmp, NM_N_EMBD);
    nm_mul_t(&s_kk, &s_a, NM_N_EMBD, 1, &s_tmp2);
    nm_decay_q15(&s_w, NM_N_EMBD, s_w15);

    nm_wkv7_recurrence(&s_q, &s_k, &s_vq, &s_tmp, &s_tmp2, s_w15, st->wkv, NM_HEAD_SIZE,
                       NM_N_EMBD, &st->wkv_step, st->wkv_step_valid, &s_y);
    st->wkv_step = nm_mul(s_k.scale, s_vq.scale);
    st->wkv_step_valid = 1;

    nm_norm_q(&s_y, &p->ln_x_weight, &p->ln_x_bias, NM_EPS_Q32_X, NM_N_HEAD, NM_N_EMBD, &s_tmp);

    nm_mul_t(&s_r, &s_k, NM_N_EMBD, 0, &s_rk);
    nm_param(&p->r_k, NM_N_EMBD, &s_tmp2);
    nm_mul_t(&s_rk, &s_tmp2, NM_N_EMBD, 1, &s_rk);
    nm_fq(&s_rk, NM_N_EMBD);
    nm_sum_head(&s_rk, NM_HEAD_SIZE, NM_N_EMBD, 0, &s_rk);
    /* sum_head(rk) 是 Q16 码、v 是未量化的对齐码，码乘积实测到 2^63.1，
     * 直接乘会 int64 回绕，所以走 128 位的融合算子（= to_fixed(a + b * c)）。 */
    nm_add_mulq(&s_tmp, &s_rk, &s_v, NM_N_EMBD, 1, &s_tmp);

    nm_mul_t(&s_tmp, &s_g, NM_N_EMBD, 1, &s_tmp);
    nm_fq(&s_tmp, NM_N_EMBD);
    nm_linear(&s_tmp, &p->output, 0, out);
}

/* 输入：层号、ln2 输出、该层状态；输出：CMix 输出。 */
static void nm_ffn(int layer, const nm_tensor *h2, nm_layer_state *st, nm_tensor *out)
{
    const nm_block *p = &nm_blocks[layer];
    static nm_tensor prev;
    if (st->ffn_prev_valid) {
        for (int j = 0; j < NM_N_EMBD; j++) prev.v[j] = st->ffn_prev[j];
        prev.scale = st->ffn_prev_scale;
        nm_sub_t(&prev, h2, NM_N_EMBD, 1, &s_tmp);
    } else {
        nm_neg_t(h2, NM_N_EMBD, &s_tmp);
    }
    nm_fq(&s_tmp, NM_N_EMBD);
    nm_addmul_param(h2, &s_tmp, &p->ffn_x_k, NM_N_EMBD, &s_tmp);

    nm_linear(&s_tmp, &p->ffn_key, 0, &s_k64);
    nm_relu_sq(&s_k64, NM_DIM_FFN, 1, &s_k64);
    nm_fq(&s_k64, NM_DIM_FFN);
    nm_linear(&s_k64, &p->ffn_value, 0, out);
}

void nm_reset(void)
{
    nm_range_error = 0;
}

void nm_state_zero(nm_layer_state *states)
{
    memset(states, 0, sizeof(nm_layer_state) * NM_N_LAYER);
}

void nm_forward_token(int token, nm_layer_state *states, int8_t *logits, nm_scale *logits_scale)
{
    int i, j;
    static nm_tensor x, h, att_out, ffn_out, v_first;
    int v_first_valid = 0;

    nm_lut_init();                   /* 幂等：第一次进来把两张 Q15 表展开进 RAM */
    nm_embed(token, &x);
    for (i = 0; i < NM_N_LAYER; i++) {
        nm_layer_state *st = &states[i];
        const nm_block *p = &nm_blocks[i];
        if (i == 0) nm_norm_q(&x, &p->ln0_weight, &p->ln0_bias, NM_EPS_Q32, 1, NM_N_EMBD, &x);
        nm_norm_q(&x, &p->ln1_weight, &p->ln1_bias, NM_EPS_Q32, 1, NM_N_EMBD, &h);
        nm_att(i, &h, &v_first, &v_first_valid, st, &att_out);
        for (j = 0; j < NM_N_EMBD; j++) st->att_prev[j] = (int8_t)h.v[j];
        st->att_prev_scale = h.scale;
        st->att_prev_valid = 1;
        nm_add_t(&x, &att_out, NM_N_EMBD, 1, &x);
        nm_fq(&x, NM_N_EMBD);

        nm_norm_q(&x, &p->ln2_weight, &p->ln2_bias, NM_EPS_Q32, 1, NM_N_EMBD, &h);
        nm_ffn(i, &h, st, &ffn_out);
        for (j = 0; j < NM_N_EMBD; j++) st->ffn_prev[j] = (int8_t)h.v[j];
        st->ffn_prev_scale = h.scale;
        st->ffn_prev_valid = 1;
        nm_add_t(&x, &ffn_out, NM_N_EMBD, 1, &x);
        nm_fq(&x, NM_N_EMBD);
    }
    nm_norm_q(&x, &nm_ln_out_weight, &nm_ln_out_bias, NM_EPS_Q32, 1, NM_N_EMBD, &x);
    nm_head_linear(&x, &nm_head_weight, logits, logits_scale);
}
