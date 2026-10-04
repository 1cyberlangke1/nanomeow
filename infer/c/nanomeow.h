/* nanomeow INT8 纯整数推理引擎（RWKV-7 x070 缩维版）对外接口。
 *
 * 输入：一个字节 token（0..255）与跨 token 的状态 `nm_layer_state`。
 * 输出：logits 的 int8 码 + 一个 scale（实值 = 码 * value(scale)，长度 NM_VOCAB）。
 * 预期行为：与 infer/ref/ 的 Python 定点参考逐位一致（G1 闸门），全程整数、无浮点、
 *           权重只有 int8，wkv state 是 int32（单位 = 当时 token 的 s_k * s_v）。
 */
#ifndef NANOMEOW_H
#define NANOMEOW_H

#include <stdint.h>

#include "nm_fixed.h"

/* 结构常量只有一处真相：由 train/scripts/export_int8.py 按 checkpoint 生成到
 * infer/model_cfg.h。引擎跟着权重头走，改维度不必手改本文件。 */
#include "../model_cfg.h"

#define NM_N_LAYER NMW_N_LAYER
#define NM_N_EMBD NMW_N_EMBD
#define NM_N_HEAD NMW_N_HEAD
#define NM_HEAD_SIZE NMW_HEAD_SIZE
#define NM_DIM_FFN NMW_DIM_FFN
#define NM_VOCAB NMW_VOCAB
#define NM_CTX_LEN NMW_CTX_LEN

/* 两张 Q15 非线性表（exp / log1p）的表长。表内容由 nm_lut.h 的位流在运行期展开，
 * 长度定义在这儿是为了让自检程序也能按同一个口径读表；gen_lut.py 会校验这一行没被改。 */
#define NM_LUT_N 257

/* 一层张量：per-row 权重给 rows 行 scale，per-tensor 常量给 1 个。
 * code_off / scale_off 不是各自独立的数组，而是统一权重池 nm_pool 里的字节偏移（uint16）——
 * 池总长生成时断言 ≤ 64 KiB，描述符因此从 12 B 缩到 8 B（109 个描述符共省 436 B Flash）。
 * code_off 处是 int8 码；idx_bits > 0 时它指向「去重后的唯一行池」，逻辑第 row 行要先过索引表
 * （索引表紧跟在 scale 位流末尾，见 nm_row_codes）。scale_off 处是位流，口径见 gen_weights.py：
 *   per-tensor：4 B 小端，(尾数 23 位) << 7 | (shift + 127)；
 *   per-row   ：[尾数 23 位打包][1 B 基线 shift][1 B 增量位宽 w][增量 w 位打包][索引 idx_bits 位打包]。
 * 尾数最高位恒 1、每张量的 shift 又几乎不变，这两条前提让 scale 表从 6,516 B 压到约 5.5 KB；
 * 行去重再把权重码省约 1.5 KB。两项都是无损的，运行期按需解出当前行的那一份。
 * rows/cols/per_row/idx_bits 收成位域 —— 四项合计仍是一个 32 位字。
 * rows/cols 上限 511，超出时生成器会先报错，不会静默截断。 */
typedef struct {
    uint16_t code_off;
    uint16_t scale_off;
    uint32_t rows : 9;
    uint32_t cols : 9;
    uint32_t per_row : 1;
    uint32_t idx_bits : 4;
} nm_mat;

/* 权重池：所有张量的 int8 码与 scale 位流首尾相接拼成的一块常量，描述符里的偏移都相对它。
 * 生成器会断言总长 ≤ 65535（当前约 46.6 KB），装不下就直接报错，不会静默截断。 */
extern const uint8_t nm_pool[];

typedef struct {
    nm_mat ln0_weight, ln0_bias;                 /* 只有第 0 层有 ln0 */
    nm_mat ln1_weight, ln1_bias, ln2_weight, ln2_bias;
    nm_mat x_r, x_w, x_k, x_v, x_a, x_g;
    nm_mat w1, w2, w0;
    nm_mat a1, a2, a0;
    nm_mat v1, v2, v0;
    nm_mat g1, g2;
    nm_mat k_k, k_a, r_k;
    nm_mat receptance, key, value, output;
    nm_mat ln_x_weight, ln_x_bias;
    nm_mat ffn_x_k, ffn_key, ffn_value;
} nm_block;

extern const nm_block nm_blocks[NM_N_LAYER];
extern const nm_mat nm_emb_weight, nm_ln_out_weight, nm_ln_out_bias, nm_head_weight;

/* 展开后的两张 Q15 表（放 RAM）：逐项与 infer/ref/nonlinear.py 的 EXP_LUT / LOG1P_LUT 相同。
 * 原样存要 1,028 B Flash，现在存的是 nm_lut.h 里每步 2 bit 的二阶差分位流（约 138 B）。 */
extern uint16_t nm_exp_lut[NM_LUT_N];
extern uint16_t nm_log1p_lut[NM_LUT_N];

/* 一层的跨 token 状态。att_prev / ffn_prev 是上一 token 的 ln1 / ln2 输出（int8 码 + scale），
 * 首个 token 前 valid = 0（对应训练侧 ZeroPad2d 的「首行视作 0」）；
 * wkv 是 [n_head][head_size][head_size] 的 int32，单位是 wkv_step（首个 token 前 valid = 0）。 */
typedef struct {
    int8_t att_prev[NM_N_EMBD];
    nm_scale att_prev_scale;
    int att_prev_valid;
    int8_t ffn_prev[NM_N_EMBD];
    nm_scale ffn_prev_scale;
    int ffn_prev_valid;
    int32_t wkv[NM_N_HEAD][NM_HEAD_SIZE][NM_HEAD_SIZE];
    nm_scale wkv_step;
    int wkv_step_valid;
} nm_layer_state;

/* 输入：无；输出：无。预期行为：清零状态并把越界哨兵复位，跑一批 token 前调用一次。 */
void nm_reset(void);

/* 输入：无；输出：无。预期行为：把 nm_lut.h 的两张 Q15 表从二阶差分位流展开进 RAM，
 * 只展开一次（幂等）；nm_forward_token 每次进来会先调它，自检程序也可以直接调来核对表内容。 */
void nm_lut_init(void);

/* 输入：状态数组（长度 NM_N_LAYER）；输出：无。预期行为：把状态置成「首个 token 之前」。 */
void nm_state_zero(nm_layer_state *states);

/* 输入：token（0..255）、状态数组；输出：logits 的 int8 码与 scale（就地更新状态）。
 * 预期行为：与 infer/ref/model.py 的 Int8Model.forward_token 逐位一致。 */
void nm_forward_token(int token, nm_layer_state *states,
                      int8_t *logits, nm_scale *logits_scale);

#endif
