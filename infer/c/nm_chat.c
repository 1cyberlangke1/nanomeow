/* nanomeow C 引擎的主机端生成入口：只用于对拍与演示，上板时把回调换成 OLED / 串口。
 *
 * 输入（stdin）：prompt 的原始字节（可含多字节 UTF-8）。
 * 输入（argv）：max_new_tokens、rep_penalty_q16、penalty_window，都可省（默认 200 / 65536 / 0）。
 * 输出（stdout）：生成出来的完整字符（原始字节，不加任何包装），方便逐字节对拍。
 * 输出（stderr）：R <nm_range_error> 与 H <hit_stop>，供测试断言没有定点越界。
 * 预期行为：全程整数、无浮点；输出的每个字节都属于某个完整字符。
 */
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
/* Windows 的文本模式会把 stdin 里的 0x1A 当 EOF、把 stdout 的 \n 翻成 \r\n，
 * 而这里跑的是字节流（token 恒在 0..255），必须切二进制，否则输入被截断、输出被改写。 */
#include <fcntl.h>
#include <io.h>
#define NM_SET_BINARY(fp) _setmode(_fileno(fp), _O_BINARY)
#else
#define NM_SET_BINARY(fp) ((void)0)
#endif

#include "nm_gen.h"

/* 输入：上下文（未用）、字节、长度；输出：无。预期行为：把完整字符原样写到 stdout。 */
static void emit_stdout(void *ctx, const uint8_t *bytes, int n)
{
    (void)ctx;
    fwrite(bytes, 1, (size_t)n, stdout);
}

int main(int argc, char **argv)
{
    static uint8_t prompt[NM_CTX_LEN * 4];
    static nm_layer_state states[NM_N_LAYER];
    nm_gen_cfg cfg;
    int prompt_len = 0, c, hit = 0;

    NM_SET_BINARY(stdin);
    NM_SET_BINARY(stdout);

    cfg.max_new_tokens = argc > 1 ? atoi(argv[1]) : 200;
    cfg.rep_penalty_q16 = (int32_t)(argc > 2 ? strtol(argv[2], NULL, 10) : 65536);
    cfg.penalty_window = argc > 3 ? atoi(argv[3]) : 0;

    while (prompt_len < (int)sizeof(prompt) && (c = getchar()) != EOF)
        prompt[prompt_len++] = (uint8_t)c;

    nm_generate(states, prompt, prompt_len, &cfg, emit_stdout, NULL, &hit);
    fflush(stdout);
    fprintf(stderr, "R %d\nH %d\n", nm_range_error, hit);
    return 0;
}
