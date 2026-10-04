/* 增量 UTF-8 解码器的主机端对拍入口：stdin 读原始字节，stdout 写「凑成完整字符」的字节。
 *
 * 输出（stderr）：`D <push 阶段丢掉的字节数> P <flush 前压在缓冲里的字节数>`，
 * 供测试与 Python 参考的 dropped_bytes / pending_bytes 对齐。
 * 预期行为：与 train/src/tokenizer.py 的 UTF8StreamDecoder 在任意字节流上输出一致。
 */
#include <stdio.h>

#ifdef _WIN32
/* 与 nm_chat.c 同理：字节流要绕开 Windows 文本模式的 0x1A=EOF 与 \n -> \r\n 改写。 */
#include <fcntl.h>
#include <io.h>
#define NM_SET_BINARY(fp) _setmode(_fileno(fp), _O_BINARY)
#else
#define NM_SET_BINARY(fp) ((void)0)
#endif

#include "nm_utf8.h"

int main(void)
{
    nm_utf8_decoder d;
    uint8_t out[4];
    int c, pending;

    NM_SET_BINARY(stdin);
    NM_SET_BINARY(stdout);

    nm_utf8_init(&d);
    while ((c = getchar()) != EOF) {
        int n = nm_utf8_push(&d, (uint8_t)c, out);
        if (n > 0) fwrite(out, 1, (size_t)n, stdout);
    }
    pending = d.len;
    nm_utf8_flush(&d);
    fflush(stdout);
    fprintf(stderr, "D %u P %d\n", (unsigned)(d.dropped - (uint32_t)pending), pending);
    return 0;
}
