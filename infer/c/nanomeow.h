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

#define NM_N_LAYER 3
#define NM_N_EMBD 32
#define NM_N_HEAD 4
#define NM_HEAD_SIZE 8
#define NM_DIM_FFN 64
#define NM_VOCAB 256
#define NM_CTX_LEN 512

/* 一层张量：per-row 权重给 rows 行 scale，per-tensor 常量给 1 个。 */
typedef struct {
    const int8_t *codes;
    const int32_t *mul;
    const int8_t *shift;
    int rows;
    int cols;
    int per_row;
} nm_mat;

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

/* 输入：状态数组（长度 NM_N_LAYER）；输出：无。预期行为：把状态置成「首个 token 之前」。 */
void nm_state_zero(nm_layer_state *states);

/* 输入：token（0..255）、状态数组；输出：logits 的 int8 码与 scale（就地更新状态）。
 * 预期行为：与 infer/ref/model.py 的 Int8Model.forward_token 逐位一致。 */
void nm_forward_token(int token, nm_layer_state *states,
                      int8_t *logits, nm_scale *logits_scale);

#endif