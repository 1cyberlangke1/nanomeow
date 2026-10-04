/* nanomeow C 引擎生成路径的实现：贪心选码 + CTRL 式重复惩罚 + <ETX> 停止。
 *
 * 为什么惩罚能在 int8 码上做：同一趟 logits 的 256 个码共用一个正 scale，所以「实值大小」
 * 与「码大小」同序；惩罚只是把出现过的正码按 1/p 缩、负码按 p 放。为了不引入浮点和舍入，
 * 比较用通分后的精确整数 key（见 nm_gen_key），不真的去改码。
 */
#include <string.h>

#include "nm_gen.h"

#define NM_ETX 0x03
#define NM_Q16 65536
#define NM_GEN_HIST_MAX NM_CTX_LEN

/* 上板固件把重复惩罚恒设成 1.0（关闭，见 keil_demo/User/main.c 的 g_cfg），历史表就成了死重：
 * 定义 NM_GEN_NO_REP_PENALTY 后，hist / hist_cnt 两张表（1 KB RAM）和每步的更新语句全部编掉，
 * 采样只看码值本身 —— 与「惩罚 = 1.0」时 nm_gen_key 走的分支逐位等价。
 * 注意：这个开关只在惩罚系数 <= 1.0 时可用；开着惩罚又定义它会退化成「无惩罚」。 */
#ifdef NM_GEN_NO_REP_PENALTY
#define NM_GEN_SEEN(id) 0
#define NM_GEN_PUSH(id) ((void)0)
#else
#define NM_GEN_SEEN(id) (hist_cnt[(id)] != 0)
#define NM_GEN_PUSH(id) do {                         \
        if (hist_len == cap) hist_cnt[hist[ring]]--; \
        else hist_len++;                             \
        hist[ring] = (uint8_t)(id);                  \
        hist_cnt[(id)]++;                            \
        ring = (ring + 1) % cap;                     \
    } while (0)
#endif

/* 输入：候选码、该码是否已经生成过、惩罚系数 Q16；输出：用于比大小的精确整数 key。
 * 预期行为：把「没出现过（值 = c）」「出现过且 c > 0（值 = c/p）」「出现过且 c < 0（值 = c*p）」
 *           三个式子通分到公共分母 D = p_num * p_den（p_num = pen_q16、p_den = 2^16），
 *           于是 key 分别是 c*D、c*p_den^2、c*p_num^2，比 key 就是比实值。
 *           惩罚关闭（pen_q16 <= 2^16）时三者都退化成 c * 2^32，等价于直接比码。 */
static int64_t nm_gen_key(int code, int seen, int32_t pen_q16)
{
    int64_t num = pen_q16;
    if (pen_q16 <= NM_Q16) return (int64_t)code * NM_Q16 * NM_Q16;
    if (!seen) return (int64_t)code * num * NM_Q16;
    if (code > 0) return (int64_t)code * NM_Q16 * NM_Q16;
    if (code < 0) return (int64_t)code * num * num;
    return 0;
}

int nm_generate(nm_layer_state *states, const uint8_t *prompt, int prompt_len,
                const nm_gen_cfg *cfg, nm_emit_fn emit, void *ctx, int *hit_stop)
{
    static int8_t logits[NM_VOCAB];
#ifndef NM_GEN_NO_REP_PENALTY
    static uint8_t hist[NM_GEN_HIST_MAX];      /* 已生成字节的环形历史 */
    static uint16_t hist_cnt[NM_VOCAB];        /* 每个字节在历史里的出现次数 */
#endif
    nm_scale logits_scale;
    nm_utf8_decoder dec;
    uint8_t out[4];
    int i, n = 0, stop = 0;
#ifndef NM_GEN_NO_REP_PENALTY
    int hist_len = 0, ring = 0;
    int cap = cfg->penalty_window > 0 ? cfg->penalty_window : NM_GEN_HIST_MAX;

    if (cap > NM_GEN_HIST_MAX) cap = NM_GEN_HIST_MAX;   /* 历史缓冲的物理上限 */
#endif
    nm_reset();
    nm_state_zero(states);
#ifndef NM_GEN_NO_REP_PENALTY
    memset(hist_cnt, 0, sizeof(hist_cnt));
#endif

    if (prompt_len <= 0) {
        nm_forward_token(0, states, logits, &logits_scale);   /* 空 prompt：占位字节起头 */
    } else {
        for (i = 0; i < prompt_len; i++)
            nm_forward_token(prompt[i], states, logits, &logits_scale);
    }

    nm_utf8_init(&dec);
    while (n < cfg->max_new_tokens) {
        int best = 0;
        int64_t best_key = nm_gen_key(logits[0], NM_GEN_SEEN(0), cfg->rep_penalty_q16);
        for (i = 1; i < NM_VOCAB; i++) {
            int64_t k = nm_gen_key(logits[i], NM_GEN_SEEN(i), cfg->rep_penalty_q16);
            if (k > best_key) { best_key = k; best = i; }   /* 并列取最小下标 = torch.argmax */
        }
        if (best == NM_ETX) { stop = 1; break; }
        n++;
        {
            int m = nm_utf8_push(&dec, (uint8_t)best, out);
            if (m > 0 && emit != 0) emit(ctx, out, m);
        }
        NM_GEN_PUSH(best);
        nm_forward_token(best, states, logits, &logits_scale);
    }
    nm_utf8_flush(&dec);
    if (hit_stop != 0) *hit_stop = stop;
    return n;
}
