/* nanomeow C 引擎的生成路径：逐字节自回归 + 整数重复惩罚 + <ETX> 停止 + 增量 UTF-8 解码。
 *
 * 输入：prompt 字节、生成参数、一个「只收完整字符」的输出回调。
 * 输出：生成的字节数；hit_stop 置 1 表示命中 <ETX>（0x03）提前停。
 * 预期行为：与 Python 定点参考（infer/ref/Int8Model）上同一套整数算法逐位一致；
 *           全程无浮点，交给回调的字节永远是完整 UTF-8 字符，绝不吐半截序列。
 */
#ifndef NANOMEOW_GEN_H
#define NANOMEOW_GEN_H

#include "nanomeow.h"
#include "nm_utf8.h"

/* 生成参数（全整数）：max_new_tokens = 生成上限；rep_penalty_q16 = 重复惩罚系数的 Q16 定点
 * （65536 = 1.0 = 关闭）；penalty_window = 只看最近多少个已生成字节，0 = 全部历史。 */
typedef struct {
    int max_new_tokens;
    int32_t rep_penalty_q16;
    int penalty_window;
} nm_gen_cfg;

/* 输出回调：把已经凑成完整字符的字节交给调用方（上板时写 OLED / 串口，主机端写 stdout）。 */
typedef void (*nm_emit_fn)(void *ctx, const uint8_t *bytes, int n);

/* 输入：状态数组、prompt 字节、prompt 长度、参数、回调与其上下文、hit_stop 出参；
 * 输出：生成的字节数（不含被 <ETX> 截断的那个）。
 * 预期行为：先复位状态并把 prompt 逐字节喂完，再贪心生成；命中 <ETX> 立刻停，
 *           结尾没凑齐的半截字符直接丢掉。 */
int nm_generate(nm_layer_state *states, const uint8_t *prompt, int prompt_len,
                const nm_gen_cfg *cfg, nm_emit_fn emit, void *ctx, int *hit_stop);

#endif
