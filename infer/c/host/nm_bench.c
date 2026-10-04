/* 主机端吞吐基准：只调 nm_forward_token，不碰生成路径，用来量「每个 token 要多少周期」。
 *
 * 输入（命令行）：argv[1] = 计时循环次数（默认 20000）；argv[2] = 折算 tok/s 用的主频 GHz（可选）。
 * 输出（stdout）：循环次数、总周期、周期/token、按给定主频折算的 tok/s、nm_range_error。
 * 预期行为：用 QueryThreadCycleTime 只数**本线程真正跑掉的周期**，别的进程抢核不会让数字失真
 *           （墙钟口径实测同一二进制两次能差 14%，这个口径抖动 < 1%）；
 *           先跑 64 次热身把缓存 / 分支预测带到稳态再计时；线程绑到 4 号核并提到最高优先级，
 *           尽量避开 Windows 调度抖动。nm_range_error 是引擎的溢出计数，必须为 0 才说明没饱和。
 *
 * 注意这是**主机（x86）口径**：Cortex-M3 的周期数只能在板上或用周期精确的模拟器量。
 * 这里的数字只用来横向比较同一台机器上的两版实现，不能直接当成 STM32 的 tok/s。
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

#include "nanomeow.h"

int main(int argc, char **argv)
{
    static nm_layer_state states[NM_N_LAYER];
    static int8_t logits[NM_VOCAB];
    nm_scale sc;
    int T = argc > 1 ? atoi(argv[1]) : 20000;
    double ghz = argc > 2 ? atof(argv[2]) : 0.0;
    int i;
    ULONG64 c0, c1;

    SetThreadAffinityMask(GetCurrentThread(), 4);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    nm_reset();
    nm_state_zero(states);
    for (i = 0; i < 64; i++)
        nm_forward_token((uint8_t)(i * 37 + 11), states, logits, &sc);

    QueryThreadCycleTime(GetCurrentThread(), &c0);
    for (i = 0; i < T; i++)
        nm_forward_token((uint8_t)(i * 37 + 11), states, logits, &sc);
    QueryThreadCycleTime(GetCurrentThread(), &c1);

    printf("T=%d  cycles=%llu  %.1f cyc/token", T,
           (unsigned long long)(c1 - c0), (double)(c1 - c0) / T);
    if (ghz > 0.0)
        printf("  %.1f tok/s @%.2fGHz", ghz * 1e9 * T / (double)(c1 - c0), ghz);
    printf("  range_error=%d\n", nm_range_error);
    return 0;
}
