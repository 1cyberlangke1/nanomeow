/* 自动生成，请勿手改：infer/c/gen_weights.py
 *
 * 权重来源：infer/model_weights.h（train/scripts/export_int8.py 从 checkpoint 导出）
 * 模型：3 层 RWKV-7 x070 缩维，n_embd=32，n_head=4，head_size=8，dim_ffn=32，词表 256
 *
 * 导出的权重（名字 = 训练侧参数名；字节数 = int8 码 / scale 表）：
 *   emb.weight                         8192 B /  1280 B  (per-row 256 x 32)
 *   blocks.0.ln1.weight                  32 B /     5 B  (per-tensor 32)
 *   blocks.0.ln1.bias                    32 B /     5 B  (per-tensor 32)
 *   blocks.0.ln2.weight                  32 B /     5 B  (per-tensor 32)
 *   blocks.0.ln2.bias                    32 B /     5 B  (per-tensor 32)
 *   blocks.0.ln0.weight                  32 B /     5 B  (per-tensor 32)
 *   blocks.0.ln0.bias                    32 B /     5 B  (per-tensor 32)
 *   blocks.0.att.x_r                     32 B /     5 B  (per-tensor 32)
 *   blocks.0.att.x_w                     32 B /     5 B  (per-tensor 32)
 *   blocks.0.att.x_k                     32 B /     5 B  (per-tensor 32)
 *   blocks.0.att.x_v                     32 B /     5 B  (per-tensor 32)
 *   blocks.0.att.x_a                     32 B /     5 B  (per-tensor 32)
 *   blocks.0.att.x_g                     32 B /     5 B  (per-tensor 32)
 *   blocks.0.att.w1                     256 B /    40 B  (per-row 8 x 32)
 *   blocks.0.att.w2                     256 B /   160 B  (per-row 32 x 8)
 *   blocks.0.att.w0                      32 B /     5 B  (per-tensor 32)
 *   blocks.0.att.a1                     256 B /    40 B  (per-row 8 x 32)
 *   blocks.0.att.a2                     256 B /   160 B  (per-row 32 x 8)
 *   blocks.0.att.a0                      32 B /     5 B  (per-tensor 32)
 *   blocks.0.att.v1                     256 B /    40 B  (per-row 8 x 32)
 *   blocks.0.att.v2                     256 B /   160 B  (per-row 32 x 8)
 *   blocks.0.att.v0                      32 B /     5 B  (per-tensor 32)
 *   blocks.0.att.g1                     256 B /    40 B  (per-row 8 x 32)
 *   blocks.0.att.g2                     256 B /   160 B  (per-row 32 x 8)
 *   blocks.0.att.k_k                     32 B /     5 B  (per-tensor 32)
 *   blocks.0.att.k_a                     32 B /     5 B  (per-tensor 32)
 *   blocks.0.att.r_k                     32 B /     5 B  (per-tensor 32)
 *   blocks.0.att.receptance.weight     1024 B /   160 B  (per-row 32 x 32)
 *   blocks.0.att.key.weight            1024 B /   160 B  (per-row 32 x 32)
 *   blocks.0.att.value.weight          1024 B /   160 B  (per-row 32 x 32)
 *   blocks.0.att.output.weight         1024 B /   160 B  (per-row 32 x 32)
 *   blocks.0.att.ln_x.weight             32 B /     5 B  (per-tensor 32)
 *   blocks.0.att.ln_x.bias               32 B /     5 B  (per-tensor 32)
 *   blocks.0.ffn.x_k                     32 B /     5 B  (per-tensor 32)
 *   blocks.0.ffn.key.weight            1024 B /   160 B  (per-row 32 x 32)
 *   blocks.0.ffn.value.weight          1024 B /   160 B  (per-row 32 x 32)
 *   blocks.1.ln1.weight                  32 B /     5 B  (per-tensor 32)
 *   blocks.1.ln1.bias                    32 B /     5 B  (per-tensor 32)
 *   blocks.1.ln2.weight                  32 B /     5 B  (per-tensor 32)
 *   blocks.1.ln2.bias                    32 B /     5 B  (per-tensor 32)
 *   blocks.1.att.x_r                     32 B /     5 B  (per-tensor 32)
 *   blocks.1.att.x_w                     32 B /     5 B  (per-tensor 32)
 *   blocks.1.att.x_k                     32 B /     5 B  (per-tensor 32)
 *   blocks.1.att.x_v                     32 B /     5 B  (per-tensor 32)
 *   blocks.1.att.x_a                     32 B /     5 B  (per-tensor 32)
 *   blocks.1.att.x_g                     32 B /     5 B  (per-tensor 32)
 *   blocks.1.att.w1                     256 B /    40 B  (per-row 8 x 32)
 *   blocks.1.att.w2                     256 B /   160 B  (per-row 32 x 8)
 *   blocks.1.att.w0                      32 B /     5 B  (per-tensor 32)
 *   blocks.1.att.a1                     256 B /    40 B  (per-row 8 x 32)
 *   blocks.1.att.a2                     256 B /   160 B  (per-row 32 x 8)
 *   blocks.1.att.a0                      32 B /     5 B  (per-tensor 32)
 *   blocks.1.att.v1                     256 B /    40 B  (per-row 8 x 32)
 *   blocks.1.att.v2                     256 B /   160 B  (per-row 32 x 8)
 *   blocks.1.att.v0                      32 B /     5 B  (per-tensor 32)
 *   blocks.1.att.g1                     256 B /    40 B  (per-row 8 x 32)
 *   blocks.1.att.g2                     256 B /   160 B  (per-row 32 x 8)
 *   blocks.1.att.k_k                     32 B /     5 B  (per-tensor 32)
 *   blocks.1.att.k_a                     32 B /     5 B  (per-tensor 32)
 *   blocks.1.att.r_k                     32 B /     5 B  (per-tensor 32)
 *   blocks.1.att.receptance.weight     1024 B /   160 B  (per-row 32 x 32)
 *   blocks.1.att.key.weight            1024 B /   160 B  (per-row 32 x 32)
 *   blocks.1.att.value.weight          1024 B /   160 B  (per-row 32 x 32)
 *   blocks.1.att.output.weight         1024 B /   160 B  (per-row 32 x 32)
 *   blocks.1.att.ln_x.weight             32 B /     5 B  (per-tensor 32)
 *   blocks.1.att.ln_x.bias               32 B /     5 B  (per-tensor 32)
 *   blocks.1.ffn.x_k                     32 B /     5 B  (per-tensor 32)
 *   blocks.1.ffn.key.weight            1024 B /   160 B  (per-row 32 x 32)
 *   blocks.1.ffn.value.weight          1024 B /   160 B  (per-row 32 x 32)
 *   blocks.2.ln1.weight                  32 B /     5 B  (per-tensor 32)
 *   blocks.2.ln1.bias                    32 B /     5 B  (per-tensor 32)
 *   blocks.2.ln2.weight                  32 B /     5 B  (per-tensor 32)
 *   blocks.2.ln2.bias                    32 B /     5 B  (per-tensor 32)
 *   blocks.2.att.x_r                     32 B /     5 B  (per-tensor 32)
 *   blocks.2.att.x_w                     32 B /     5 B  (per-tensor 32)
 *   blocks.2.att.x_k                     32 B /     5 B  (per-tensor 32)
 *   blocks.2.att.x_v                     32 B /     5 B  (per-tensor 32)
 *   blocks.2.att.x_a                     32 B /     5 B  (per-tensor 32)
 *   blocks.2.att.x_g                     32 B /     5 B  (per-tensor 32)
 *   blocks.2.att.w1                     256 B /    40 B  (per-row 8 x 32)
 *   blocks.2.att.w2                     256 B /   160 B  (per-row 32 x 8)
 *   blocks.2.att.w0                      32 B /     5 B  (per-tensor 32)
 *   blocks.2.att.a1                     256 B /    40 B  (per-row 8 x 32)
 *   blocks.2.att.a2                     256 B /   160 B  (per-row 32 x 8)
 *   blocks.2.att.a0                      32 B /     5 B  (per-tensor 32)
 *   blocks.2.att.v1                     256 B /    40 B  (per-row 8 x 32)
 *   blocks.2.att.v2                     256 B /   160 B  (per-row 32 x 8)
 *   blocks.2.att.v0                      32 B /     5 B  (per-tensor 32)
 *   blocks.2.att.g1                     256 B /    40 B  (per-row 8 x 32)
 *   blocks.2.att.g2                     256 B /   160 B  (per-row 32 x 8)
 *   blocks.2.att.k_k                     32 B /     5 B  (per-tensor 32)
 *   blocks.2.att.k_a                     32 B /     5 B  (per-tensor 32)
 *   blocks.2.att.r_k                     32 B /     5 B  (per-tensor 32)
 *   blocks.2.att.receptance.weight     1024 B /   160 B  (per-row 32 x 32)
 *   blocks.2.att.key.weight            1024 B /   160 B  (per-row 32 x 32)
 *   blocks.2.att.value.weight          1024 B /   160 B  (per-row 32 x 32)
 *   blocks.2.att.output.weight         1024 B /   160 B  (per-row 32 x 32)
 *   blocks.2.att.ln_x.weight             32 B /     5 B  (per-tensor 32)
 *   blocks.2.att.ln_x.bias               32 B /     5 B  (per-tensor 32)
 *   blocks.2.ffn.x_k                     32 B /     5 B  (per-tensor 32)
 *   blocks.2.ffn.key.weight            1024 B /   160 B  (per-row 32 x 32)
 *   blocks.2.ffn.value.weight          1024 B /   160 B  (per-row 32 x 32)
 *   ln_out.weight                        32 B /     5 B  (per-tensor 32)
 *   ln_out.bias                          32 B /     5 B  (per-tensor 32)
 *   head.weight                        8192 B /  1280 B  (per-row 256 x 32)
 *
 * 权重合计：码 42912 B（41.9 KiB）+ scale 表 8145 B（8.0 KiB）= 51057 B（49.9 KiB）
 * 内存占用（Flash 常驻只读表）：51057 B（49.9 KiB）
 * 内存占用（RAM）：状态 3 层 x 1124 B = 3372 B（3.3 KiB，含每层 int32 wkv 1024 B）
 *                  + 引擎工作区 24 个 int64[32] 暂存 + Q15 衰减表 = 8512 B（8.3 KiB）
 *                  = 11884 B（11.6 KiB），不含调用栈
 */

#include <stddef.h>
#include "../model_weights.h"
#include "nanomeow.h"

const nm_block nm_blocks[NM_N_LAYER] = {
    { /* layer 0 */
        {nmw_blocks_0_ln0_weight, &nmw_blocks_0_ln0_weight_mul, &nmw_blocks_0_ln0_weight_shift, 1, 32, 0},
        {nmw_blocks_0_ln0_bias, &nmw_blocks_0_ln0_bias_mul, &nmw_blocks_0_ln0_bias_shift, 1, 32, 0},
        {nmw_blocks_0_ln1_weight, &nmw_blocks_0_ln1_weight_mul, &nmw_blocks_0_ln1_weight_shift, 1, 32, 0},
        {nmw_blocks_0_ln1_bias, &nmw_blocks_0_ln1_bias_mul, &nmw_blocks_0_ln1_bias_shift, 1, 32, 0},
        {nmw_blocks_0_ln2_weight, &nmw_blocks_0_ln2_weight_mul, &nmw_blocks_0_ln2_weight_shift, 1, 32, 0},
        {nmw_blocks_0_ln2_bias, &nmw_blocks_0_ln2_bias_mul, &nmw_blocks_0_ln2_bias_shift, 1, 32, 0},
        {nmw_blocks_0_att_x_r, &nmw_blocks_0_att_x_r_mul, &nmw_blocks_0_att_x_r_shift, 1, 32, 0},
        {nmw_blocks_0_att_x_w, &nmw_blocks_0_att_x_w_mul, &nmw_blocks_0_att_x_w_shift, 1, 32, 0},
        {nmw_blocks_0_att_x_k, &nmw_blocks_0_att_x_k_mul, &nmw_blocks_0_att_x_k_shift, 1, 32, 0},
        {nmw_blocks_0_att_x_v, &nmw_blocks_0_att_x_v_mul, &nmw_blocks_0_att_x_v_shift, 1, 32, 0},
        {nmw_blocks_0_att_x_a, &nmw_blocks_0_att_x_a_mul, &nmw_blocks_0_att_x_a_shift, 1, 32, 0},
        {nmw_blocks_0_att_x_g, &nmw_blocks_0_att_x_g_mul, &nmw_blocks_0_att_x_g_shift, 1, 32, 0},
        {nmw_blocks_0_att_w1, nmw_blocks_0_att_w1_mul, nmw_blocks_0_att_w1_shift, 8, 32, 1},
        {nmw_blocks_0_att_w2, nmw_blocks_0_att_w2_mul, nmw_blocks_0_att_w2_shift, 32, 8, 1},
        {nmw_blocks_0_att_w0, &nmw_blocks_0_att_w0_mul, &nmw_blocks_0_att_w0_shift, 1, 32, 0},
        {nmw_blocks_0_att_a1, nmw_blocks_0_att_a1_mul, nmw_blocks_0_att_a1_shift, 8, 32, 1},
        {nmw_blocks_0_att_a2, nmw_blocks_0_att_a2_mul, nmw_blocks_0_att_a2_shift, 32, 8, 1},
        {nmw_blocks_0_att_a0, &nmw_blocks_0_att_a0_mul, &nmw_blocks_0_att_a0_shift, 1, 32, 0},
        {nmw_blocks_0_att_v1, nmw_blocks_0_att_v1_mul, nmw_blocks_0_att_v1_shift, 8, 32, 1},
        {nmw_blocks_0_att_v2, nmw_blocks_0_att_v2_mul, nmw_blocks_0_att_v2_shift, 32, 8, 1},
        {nmw_blocks_0_att_v0, &nmw_blocks_0_att_v0_mul, &nmw_blocks_0_att_v0_shift, 1, 32, 0},
        {nmw_blocks_0_att_g1, nmw_blocks_0_att_g1_mul, nmw_blocks_0_att_g1_shift, 8, 32, 1},
        {nmw_blocks_0_att_g2, nmw_blocks_0_att_g2_mul, nmw_blocks_0_att_g2_shift, 32, 8, 1},
        {nmw_blocks_0_att_k_k, &nmw_blocks_0_att_k_k_mul, &nmw_blocks_0_att_k_k_shift, 1, 32, 0},
        {nmw_blocks_0_att_k_a, &nmw_blocks_0_att_k_a_mul, &nmw_blocks_0_att_k_a_shift, 1, 32, 0},
        {nmw_blocks_0_att_r_k, &nmw_blocks_0_att_r_k_mul, &nmw_blocks_0_att_r_k_shift, 1, 32, 0},
        {nmw_blocks_0_att_receptance_weight, nmw_blocks_0_att_receptance_weight_mul, nmw_blocks_0_att_receptance_weight_shift, 32, 32, 1},
        {nmw_blocks_0_att_key_weight, nmw_blocks_0_att_key_weight_mul, nmw_blocks_0_att_key_weight_shift, 32, 32, 1},
        {nmw_blocks_0_att_value_weight, nmw_blocks_0_att_value_weight_mul, nmw_blocks_0_att_value_weight_shift, 32, 32, 1},
        {nmw_blocks_0_att_output_weight, nmw_blocks_0_att_output_weight_mul, nmw_blocks_0_att_output_weight_shift, 32, 32, 1},
        {nmw_blocks_0_att_ln_x_weight, &nmw_blocks_0_att_ln_x_weight_mul, &nmw_blocks_0_att_ln_x_weight_shift, 1, 32, 0},
        {nmw_blocks_0_att_ln_x_bias, &nmw_blocks_0_att_ln_x_bias_mul, &nmw_blocks_0_att_ln_x_bias_shift, 1, 32, 0},
        {nmw_blocks_0_ffn_x_k, &nmw_blocks_0_ffn_x_k_mul, &nmw_blocks_0_ffn_x_k_shift, 1, 32, 0},
        {nmw_blocks_0_ffn_key_weight, nmw_blocks_0_ffn_key_weight_mul, nmw_blocks_0_ffn_key_weight_shift, 32, 32, 1},
        {nmw_blocks_0_ffn_value_weight, nmw_blocks_0_ffn_value_weight_mul, nmw_blocks_0_ffn_value_weight_shift, 32, 32, 1},
    },
    { /* layer 1 */
        {NULL, NULL, NULL, 0, 0, 0},
        {NULL, NULL, NULL, 0, 0, 0},
        {nmw_blocks_1_ln1_weight, &nmw_blocks_1_ln1_weight_mul, &nmw_blocks_1_ln1_weight_shift, 1, 32, 0},
        {nmw_blocks_1_ln1_bias, &nmw_blocks_1_ln1_bias_mul, &nmw_blocks_1_ln1_bias_shift, 1, 32, 0},
        {nmw_blocks_1_ln2_weight, &nmw_blocks_1_ln2_weight_mul, &nmw_blocks_1_ln2_weight_shift, 1, 32, 0},
        {nmw_blocks_1_ln2_bias, &nmw_blocks_1_ln2_bias_mul, &nmw_blocks_1_ln2_bias_shift, 1, 32, 0},
        {nmw_blocks_1_att_x_r, &nmw_blocks_1_att_x_r_mul, &nmw_blocks_1_att_x_r_shift, 1, 32, 0},
        {nmw_blocks_1_att_x_w, &nmw_blocks_1_att_x_w_mul, &nmw_blocks_1_att_x_w_shift, 1, 32, 0},
        {nmw_blocks_1_att_x_k, &nmw_blocks_1_att_x_k_mul, &nmw_blocks_1_att_x_k_shift, 1, 32, 0},
        {nmw_blocks_1_att_x_v, &nmw_blocks_1_att_x_v_mul, &nmw_blocks_1_att_x_v_shift, 1, 32, 0},
        {nmw_blocks_1_att_x_a, &nmw_blocks_1_att_x_a_mul, &nmw_blocks_1_att_x_a_shift, 1, 32, 0},
        {nmw_blocks_1_att_x_g, &nmw_blocks_1_att_x_g_mul, &nmw_blocks_1_att_x_g_shift, 1, 32, 0},
        {nmw_blocks_1_att_w1, nmw_blocks_1_att_w1_mul, nmw_blocks_1_att_w1_shift, 8, 32, 1},
        {nmw_blocks_1_att_w2, nmw_blocks_1_att_w2_mul, nmw_blocks_1_att_w2_shift, 32, 8, 1},
        {nmw_blocks_1_att_w0, &nmw_blocks_1_att_w0_mul, &nmw_blocks_1_att_w0_shift, 1, 32, 0},
        {nmw_blocks_1_att_a1, nmw_blocks_1_att_a1_mul, nmw_blocks_1_att_a1_shift, 8, 32, 1},
        {nmw_blocks_1_att_a2, nmw_blocks_1_att_a2_mul, nmw_blocks_1_att_a2_shift, 32, 8, 1},
        {nmw_blocks_1_att_a0, &nmw_blocks_1_att_a0_mul, &nmw_blocks_1_att_a0_shift, 1, 32, 0},
        {nmw_blocks_1_att_v1, nmw_blocks_1_att_v1_mul, nmw_blocks_1_att_v1_shift, 8, 32, 1},
        {nmw_blocks_1_att_v2, nmw_blocks_1_att_v2_mul, nmw_blocks_1_att_v2_shift, 32, 8, 1},
        {nmw_blocks_1_att_v0, &nmw_blocks_1_att_v0_mul, &nmw_blocks_1_att_v0_shift, 1, 32, 0},
        {nmw_blocks_1_att_g1, nmw_blocks_1_att_g1_mul, nmw_blocks_1_att_g1_shift, 8, 32, 1},
        {nmw_blocks_1_att_g2, nmw_blocks_1_att_g2_mul, nmw_blocks_1_att_g2_shift, 32, 8, 1},
        {nmw_blocks_1_att_k_k, &nmw_blocks_1_att_k_k_mul, &nmw_blocks_1_att_k_k_shift, 1, 32, 0},
        {nmw_blocks_1_att_k_a, &nmw_blocks_1_att_k_a_mul, &nmw_blocks_1_att_k_a_shift, 1, 32, 0},
        {nmw_blocks_1_att_r_k, &nmw_blocks_1_att_r_k_mul, &nmw_blocks_1_att_r_k_shift, 1, 32, 0},
        {nmw_blocks_1_att_receptance_weight, nmw_blocks_1_att_receptance_weight_mul, nmw_blocks_1_att_receptance_weight_shift, 32, 32, 1},
        {nmw_blocks_1_att_key_weight, nmw_blocks_1_att_key_weight_mul, nmw_blocks_1_att_key_weight_shift, 32, 32, 1},
        {nmw_blocks_1_att_value_weight, nmw_blocks_1_att_value_weight_mul, nmw_blocks_1_att_value_weight_shift, 32, 32, 1},
        {nmw_blocks_1_att_output_weight, nmw_blocks_1_att_output_weight_mul, nmw_blocks_1_att_output_weight_shift, 32, 32, 1},
        {nmw_blocks_1_att_ln_x_weight, &nmw_blocks_1_att_ln_x_weight_mul, &nmw_blocks_1_att_ln_x_weight_shift, 1, 32, 0},
        {nmw_blocks_1_att_ln_x_bias, &nmw_blocks_1_att_ln_x_bias_mul, &nmw_blocks_1_att_ln_x_bias_shift, 1, 32, 0},
        {nmw_blocks_1_ffn_x_k, &nmw_blocks_1_ffn_x_k_mul, &nmw_blocks_1_ffn_x_k_shift, 1, 32, 0},
        {nmw_blocks_1_ffn_key_weight, nmw_blocks_1_ffn_key_weight_mul, nmw_blocks_1_ffn_key_weight_shift, 32, 32, 1},
        {nmw_blocks_1_ffn_value_weight, nmw_blocks_1_ffn_value_weight_mul, nmw_blocks_1_ffn_value_weight_shift, 32, 32, 1},
    },
    { /* layer 2 */
        {NULL, NULL, NULL, 0, 0, 0},
        {NULL, NULL, NULL, 0, 0, 0},
        {nmw_blocks_2_ln1_weight, &nmw_blocks_2_ln1_weight_mul, &nmw_blocks_2_ln1_weight_shift, 1, 32, 0},
        {nmw_blocks_2_ln1_bias, &nmw_blocks_2_ln1_bias_mul, &nmw_blocks_2_ln1_bias_shift, 1, 32, 0},
        {nmw_blocks_2_ln2_weight, &nmw_blocks_2_ln2_weight_mul, &nmw_blocks_2_ln2_weight_shift, 1, 32, 0},
        {nmw_blocks_2_ln2_bias, &nmw_blocks_2_ln2_bias_mul, &nmw_blocks_2_ln2_bias_shift, 1, 32, 0},
        {nmw_blocks_2_att_x_r, &nmw_blocks_2_att_x_r_mul, &nmw_blocks_2_att_x_r_shift, 1, 32, 0},
        {nmw_blocks_2_att_x_w, &nmw_blocks_2_att_x_w_mul, &nmw_blocks_2_att_x_w_shift, 1, 32, 0},
        {nmw_blocks_2_att_x_k, &nmw_blocks_2_att_x_k_mul, &nmw_blocks_2_att_x_k_shift, 1, 32, 0},
        {nmw_blocks_2_att_x_v, &nmw_blocks_2_att_x_v_mul, &nmw_blocks_2_att_x_v_shift, 1, 32, 0},
        {nmw_blocks_2_att_x_a, &nmw_blocks_2_att_x_a_mul, &nmw_blocks_2_att_x_a_shift, 1, 32, 0},
        {nmw_blocks_2_att_x_g, &nmw_blocks_2_att_x_g_mul, &nmw_blocks_2_att_x_g_shift, 1, 32, 0},
        {nmw_blocks_2_att_w1, nmw_blocks_2_att_w1_mul, nmw_blocks_2_att_w1_shift, 8, 32, 1},
        {nmw_blocks_2_att_w2, nmw_blocks_2_att_w2_mul, nmw_blocks_2_att_w2_shift, 32, 8, 1},
        {nmw_blocks_2_att_w0, &nmw_blocks_2_att_w0_mul, &nmw_blocks_2_att_w0_shift, 1, 32, 0},
        {nmw_blocks_2_att_a1, nmw_blocks_2_att_a1_mul, nmw_blocks_2_att_a1_shift, 8, 32, 1},
        {nmw_blocks_2_att_a2, nmw_blocks_2_att_a2_mul, nmw_blocks_2_att_a2_shift, 32, 8, 1},
        {nmw_blocks_2_att_a0, &nmw_blocks_2_att_a0_mul, &nmw_blocks_2_att_a0_shift, 1, 32, 0},
        {nmw_blocks_2_att_v1, nmw_blocks_2_att_v1_mul, nmw_blocks_2_att_v1_shift, 8, 32, 1},
        {nmw_blocks_2_att_v2, nmw_blocks_2_att_v2_mul, nmw_blocks_2_att_v2_shift, 32, 8, 1},
        {nmw_blocks_2_att_v0, &nmw_blocks_2_att_v0_mul, &nmw_blocks_2_att_v0_shift, 1, 32, 0},
        {nmw_blocks_2_att_g1, nmw_blocks_2_att_g1_mul, nmw_blocks_2_att_g1_shift, 8, 32, 1},
        {nmw_blocks_2_att_g2, nmw_blocks_2_att_g2_mul, nmw_blocks_2_att_g2_shift, 32, 8, 1},
        {nmw_blocks_2_att_k_k, &nmw_blocks_2_att_k_k_mul, &nmw_blocks_2_att_k_k_shift, 1, 32, 0},
        {nmw_blocks_2_att_k_a, &nmw_blocks_2_att_k_a_mul, &nmw_blocks_2_att_k_a_shift, 1, 32, 0},
        {nmw_blocks_2_att_r_k, &nmw_blocks_2_att_r_k_mul, &nmw_blocks_2_att_r_k_shift, 1, 32, 0},
        {nmw_blocks_2_att_receptance_weight, nmw_blocks_2_att_receptance_weight_mul, nmw_blocks_2_att_receptance_weight_shift, 32, 32, 1},
        {nmw_blocks_2_att_key_weight, nmw_blocks_2_att_key_weight_mul, nmw_blocks_2_att_key_weight_shift, 32, 32, 1},
        {nmw_blocks_2_att_value_weight, nmw_blocks_2_att_value_weight_mul, nmw_blocks_2_att_value_weight_shift, 32, 32, 1},
        {nmw_blocks_2_att_output_weight, nmw_blocks_2_att_output_weight_mul, nmw_blocks_2_att_output_weight_shift, 32, 32, 1},
        {nmw_blocks_2_att_ln_x_weight, &nmw_blocks_2_att_ln_x_weight_mul, &nmw_blocks_2_att_ln_x_weight_shift, 1, 32, 0},
        {nmw_blocks_2_att_ln_x_bias, &nmw_blocks_2_att_ln_x_bias_mul, &nmw_blocks_2_att_ln_x_bias_shift, 1, 32, 0},
        {nmw_blocks_2_ffn_x_k, &nmw_blocks_2_ffn_x_k_mul, &nmw_blocks_2_ffn_x_k_shift, 1, 32, 0},
        {nmw_blocks_2_ffn_key_weight, nmw_blocks_2_ffn_key_weight_mul, nmw_blocks_2_ffn_key_weight_shift, 32, 32, 1},
        {nmw_blocks_2_ffn_value_weight, nmw_blocks_2_ffn_value_weight_mul, nmw_blocks_2_ffn_value_weight_shift, 32, 32, 1},
    },
};

const nm_mat nm_emb_weight = {nmw_emb_weight, nmw_emb_weight_mul, nmw_emb_weight_shift, 256, 32, 1};
const nm_mat nm_ln_out_weight = {nmw_ln_out_weight, &nmw_ln_out_weight_mul, &nmw_ln_out_weight_shift, 1, 32, 0};
const nm_mat nm_ln_out_bias = {nmw_ln_out_bias, &nmw_ln_out_bias_mul, &nmw_ln_out_bias_shift, 1, 32, 0};
const nm_mat nm_head_weight = {nmw_head_weight, nmw_head_weight_mul, nmw_head_weight_shift, 256, 32, 1};
