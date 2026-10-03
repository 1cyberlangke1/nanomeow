/* nanomeow C 引擎的逐 token 对拍入口。
 *
 * 输入（stdin）：第一行 token 个数 n，之后 n 个 0..255 的整数（空格/换行分隔）。
 * 输出（stdout）：每个 token 两行 —— `S <mul> <shift>` 是该步 logits 的 scale，
 *                 下一行是 256 个 logits 码（空格分隔）；最后一行 `R <nm_range_error>`。
 * 预期行为：同一串 token 下与 infer/ref/model.py 的 Int8Model 逐位一致（G1 闸门），
 *           退出码 0 表示输入合法、全程没有触发定点溢出哨兵。
 */
#include <stdio.h>

#include "nanomeow.h"

int main(void)
{
    int n = 0, i, j;
    static int tokens[NM_CTX_LEN];
    static nm_layer_state states[NM_N_LAYER];
    static int8_t logits[NM_VOCAB];
    nm_scale sc;

    if (scanf("%d", &n) != 1 || n < 0 || n > NM_CTX_LEN) return 2;
    for (i = 0; i < n; i++) {
        if (scanf("%d", &tokens[i]) != 1) return 2;
        if (tokens[i] < 0 || tokens[i] > 255) return 2;
    }

    nm_reset();
    nm_state_zero(states);
    for (i = 0; i < n; i++) {
        nm_forward_token(tokens[i], states, logits, &sc);
        printf("S %d %d\n", sc.m, sc.e);
        for (j = 0; j < NM_VOCAB; j++) printf("%d%c", logits[j], j + 1 == NM_VOCAB ? '\n' : ' ');
    }
    printf("R %d\n", nm_range_error);
    return 0;
}
