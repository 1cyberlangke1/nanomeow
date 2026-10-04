/* nanomeow 上板入口：USART1 收一行 → 模型生成 → SSD1306 显示（同一份字节回显到串口），
 * 每轮生成完在屏上显示这次生成的**模型速度**（tokens/s）。
 *
 * 输入：USART1（PA10，115200 8N1）收字节，'\n' 结束一行；输出：SSD1306 显示 + USART1（PA9）回显。
 * 预期行为：上电把系统时钟配到 72 MHz（HSE 8 MHz × 9）、点屏，并在屏幕正中显示一行英文
 *           「nanomeow ready」自证固件已经跑起来；然后循环
 *           「等一行输入 → 清屏 → 第一行显示输入本身 → 第二行起边生成边刷屏 → 最后显示 tps」。
 *           清屏在**收到输入之后**：这样 tps 画完会一直留在屏上等下一次输入，而不是刚画完就被擦掉。
 *           屏上不显示 user: / bot: 这类前缀，只有输入、输出、速度三样。
 *           提示词严格按训练模板拼：user:<内容>\nbot:（冒号后没有空格）。全程无浮点。
 * 接线：OLED 的 SCL / SDA 引脚与串口波特率都在 User/config.h（唯一改线的地方），
 *       工程通过 -DNM_BOARD_CONFIG="config.h" 把它交给 nm_board.h。
 * 说明：Keil 工程由 Start/startup_nanomeow.s 提供向量表（宏 NM_VECTORS_IN_STARTUP）；
 *       build_firmware.py 的 clang 链路没有 startup 文件，由本文件自带向量表与 Reset_Handler。
 */
#include <string.h>

#include "nm_board.h"
#include "nm_gen.h"
#include "nm_oled.h"
#include "nm_stm32f103.h"

#define NM_LINE_MAX 256                /* 一行输入的上限（约 85 个汉字）；超出部分丢掉，不溢出 */
#define NM_PROMPT_MAX (NM_LINE_MAX + 16)
#define NM_CYC_PER_MS 72000u           /* 72 MHz 下 1 ms 的周期数 */

static nm_layer_state g_states[NM_N_LAYER];
static uint8_t g_line[NM_LINE_MAX];
static uint8_t g_prompt[NM_PROMPT_MAX];
/* 与主机演示同一口径：最多 128 个字节、重复惩罚关闭（65536 = 1.0）、窗口 0。 */
static const nm_gen_cfg g_cfg = { 128, 65536, 0 };
/* 生成期间花在「回显 + 刷屏」上的周期数，算模型速度时要减掉（只算模型本身）。 */
static uint32_t g_emit_cyc;

/* 输入：无；输出：无。预期行为：HSE 8 MHz × PLL9 = 72 MHz；Flash 2 等待周期；APB1 /2、APB2 /1。 */
static void nm_sys_clock_72m(void)
{
    NM_FLASH_ACR = 0x12u;                        /* LATENCY = 2 个等待周期 + 预取 */
    NM_RCC_CR |= 0x00010000u;                    /* HSEON */
    while (!(NM_RCC_CR & 0x00020000u)) { }       /* 等 HSERDY */
    NM_RCC_CFGR = 0x001D0400u;                   /* PLLSRC = HSE、PLLMUL = 9、PPRE1 = /2、SW = HSI */
    NM_RCC_CR |= 0x01000000u;                    /* PLLON */
    while (!(NM_RCC_CR & 0x02000000u)) { }       /* 等 PLLRDY */
    NM_RCC_CFGR |= 0x00000002u;                  /* SW = PLL */
    while ((NM_RCC_CFGR & 0x0000000Cu) != 0x00000008u) { }   /* 等 SWS = PLL */
}

/* 输入：无；输出：无。预期行为：开 DWT 的周期计数器（DEMCR.TRCENA → DWT_CTRL.CYCCNTENA），清零起算。
 * 为什么用 DWT：Cortex-M3 自带的 32 位周期计数器，不需要中断、不占定时器，读一次就是一个周期数。 */
static void nm_cyc_init(void)
{
    NM_DEMCR |= 0x01000000u;
    NM_DWT_CYCCNT = 0;
    NM_DWT_CTRL |= 1u;
}

/* 输入：无；输出：无。预期行为：PA9 = 复用推挽输出、PA10 = 浮空输入，USART1 开 115200 收发。 */
static void nm_uart_init(void)
{
    NM_RCC_APB2ENR |= NM_APB2_IOPAEN | NM_APB2_USART1EN;
    NM_GPIOA_CRH = (NM_GPIOA_CRH & ~0x00000FF0u) | 0x000004B0u;   /* PA9 = 0xB、PA10 = 0x4 */
    NM_USART1_BRR = NM_UART_BRR;
    NM_USART1_CR1 = 0x200Cu;                                      /* UE | TE | RE */
}

/* 输入：一个字节；输出：无。预期行为：等发送寄存器空再写，阻塞式。 */
static void nm_uart_putc(uint8_t c)
{
    while (!(NM_USART1_SR & 0x80u)) { }   /* 等 TXE */
    NM_USART1_DR = c;
}

/* 输入：无；输出：收到的字节。预期行为：阻塞等到收到为止。 */
static uint8_t nm_uart_getc(void)
{
    while (!(NM_USART1_SR & 0x20u)) { }   /* 等 RXNE */
    return (uint8_t)NM_USART1_DR;
}

/* 输入：毫秒数；输出：无。预期行为：72 MHz 下约 7200 次空转算 1 ms，只用于上电等屏、不要求精确。 */
static void nm_delay_ms(int ms)
{
    volatile uint32_t n = (uint32_t)ms * 7200u;
    while (n--) { }
}

/* 输入：缓冲与容量；输出：这一行的字节数。预期行为：阻塞读到 '\n'；忽略 '\r'；超容量丢弃多余字节。 */
static int nm_uart_read_line(uint8_t *buf, int cap)
{
    int n = 0;

    for (;;) {
        uint8_t c = nm_uart_getc();
        if (c == '\r') continue;
        if (c == '\n') break;
        if (n < cap) buf[n++] = c;
    }
    return n;
}

/* 输入：以 '\0' 结尾的 ASCII 串；输出：无。预期行为：逐字符画到屏上（ASCII 一定是完整字符）。 */
static void nm_oled_ascii(const char *s)
{
    while (*s != '\0') nm_oled_putc((uint32_t)(uint8_t)*s++);
}

/* 上电先显示的两行：屏幕正中、纯英文，用来自证「固件已经在跑」。收到第一行输入时被清掉。
 * 一行只有 NM_OLED_COLS 格，"nanomeow ready" 一行放不下，所以拆成两行。 */
static const char nm_ready_l1[] = "nanomeow";
static const char nm_ready_l2[] = "ready";

/* 输入：无；输出：无。预期行为：把两行横幅水平居中、在正文区里垂直居中画出来，然后整屏刷新一次。 */
static void nm_show_ready(void)
{
    int l1 = 0, l2 = 0;
    int top = (NM_OLED_TEXT_PAGES - 2 * NM_OLED_LINE_PAGES) / 2;   /* 两行横幅垂直居中 */

    while (nm_ready_l1[l1] != '\0') l1++;
    while (nm_ready_l2[l2] != '\0') l2++;
    nm_oled_goto(top, (NM_OLED_COLS - l1) / 2);
    nm_oled_ascii(nm_ready_l1);
    nm_oled_goto(top + NM_OLED_LINE_PAGES, (NM_OLED_COLS - l2) / 2);
    nm_oled_ascii(nm_ready_l2);
    nm_oled_flush();
}

/* 输入：一行 UTF-8 字节与长度；输出：无。
 * 预期行为：按首字节长度切出完整字符再交给 nm_oled_put_utf8；结尾没凑齐的半截序列不画。 */
static void nm_oled_utf8(const uint8_t *s, int n)
{
    int i = 0;

    while (i < n) {
        int need = nm_utf8_seq_len(s[i]);
        if (need == 0) need = 1;          /* 非法首字节：交给 nm_oled_put_utf8 画兜底方框 */
        if (i + need > n) break;
        nm_oled_put_utf8(s + i, need);
        i += need;
    }
}

/* 输入：上下文（未用）、一个完整字符的字节与长度；输出：无。
 * 预期行为：同一份字节既回显到串口，也画到屏上并只刷脏页 —— 生成过程中就能看到字出来；
 *           这一段时间（回显 + 刷屏）记进 g_emit_cyc，最后从总周期里扣掉，tps 只算模型。 */
static void nm_fw_emit(void *ctx, const uint8_t *bytes, int n)
{
    uint32_t t0 = NM_DWT_CYCCNT;
    int i;

    (void)ctx;
    for (i = 0; i < n; i++) nm_uart_putc(bytes[i]);
    nm_oled_put_utf8(bytes, n);
    nm_oled_flush();
    g_emit_cyc += NM_DWT_CYCCNT - t0;
}

/* 输入：本次生成的 token 数、模型占用的周期数；输出：无。
 * 预期行为：把周期换成毫秒再算 tokens/s（保留一位小数），**右对齐画在最后一页的右下角** ——
 *           位置固定、不跟正文一起流动；正文只占前 NM_OLED_TEXT_PAGES 页，盖不到它。
 *           拿不到周期（模拟器里 DWT 不计数）或不足 1 ms 时不画，避免除零。 */
static void nm_show_tps(int tokens, uint32_t cycles)
{
    uint32_t ms = cycles / NM_CYC_PER_MS;
    uint32_t x10, whole, v;
    int digits = 1, col;

    if (ms == 0) return;
    x10 = (uint32_t)tokens * 10000u / ms;   /* tokens <= 128，乘 10000 不会溢出 */
    whole = x10 / 10u;
    for (v = whole; v >= 10u; v /= 10u) digits++;
    col = NM_OLED_COLS - (6 + digits);      /* "tps:" + 整数 + "." + 一位小数，右对齐 */
    nm_oled_goto(NM_OLED_PAGES - NM_OLED_LINE_PAGES, col < 0 ? 0 : col);
    nm_oled_ascii("tps:");
    nm_oled_putu(whole);
    nm_oled_putc('.');
    nm_oled_putu(x10 % 10u);
}

/* 显式原型：Keil 的默认告警集里有 -Wmissing-prototypes，定义前先声明。 */
int main(void);

int main(void)
{
    nm_sys_clock_72m();
    nm_uart_init();
    nm_oled_port_init();
    nm_cyc_init();
    nm_delay_ms(100);              /* SSD1306 上电后要等电源稳定才收命令 */
    nm_oled_init();

    nm_show_ready();               /* 上电就先在屏中间显示 ready，证明固件跑起来了 */

    for (;;) {
        int n, p = 0, hit = 0, gen;
        uint32_t c0, e0, cyc;

        n = nm_uart_read_line(g_line, NM_LINE_MAX);
        if (n <= 0) continue;

        nm_oled_clear();                  /* 等输入期间留着上一轮的 tps，收到新输入才清屏 */
        nm_oled_utf8(g_line, n);          /* 屏上第一行就是用户输入，不加任何前缀 */
        nm_oled_newline();
        nm_oled_flush();

        /* 下面这两行才是**模型提示词**，模板必须严格是 user:<内容>\nbot:（冒号后没有空格），
         * 和屏上画什么无关 —— 屏上不显示前缀，提示词里不能少。 */
        memcpy(g_prompt + p, "user:", 5); p += 5;
        memcpy(g_prompt + p, g_line, (size_t)n); p += n;
        g_prompt[p++] = '\n';
        memcpy(g_prompt + p, "bot:", 4); p += 4;

        c0 = NM_DWT_CYCCNT;
        e0 = g_emit_cyc;
        gen = nm_generate(g_states, g_prompt, p, &g_cfg, nm_fw_emit, 0, &hit);
        cyc = (NM_DWT_CYCCNT - c0) - (g_emit_cyc - e0);

        nm_uart_putc('\n');
        nm_oled_newline();
        nm_show_tps(gen, cyc);
        nm_oled_flush();
    }
}

#ifndef NM_VECTORS_IN_STARTUP
/* 上板入口：复位后直接进 main，跑完不回。中断向量表只有栈顶与复位两个表项（本固件不开中断）。
 * Keil 工程由 Start/startup_nanomeow.s 提供向量表并定义 NM_VECTORS_IN_STARTUP，这段就不参与编译。 */
void Reset_Handler(void)
{
    (void)main();
    for (;;) { }
}

/* 栈顶 = RAM 末尾（20 KiB 的上界），栈向下生长。
 * 注意：向量表第 0 项必须是**这个数值本身**，不是它的地址 —— CPU 复位时直接把这一项装进 SP，
 * 写成 `&g_stack_top` 会把 SP 设成 Flash 里的地址，真机第一次压栈就硬 fault（模拟器实测抓到过）。 */
#define NM_STACK_TOP 0x20005000u

__attribute__((section(".isr_vector"), used))
static const void *g_vectors[48] = {
    (const void *)NM_STACK_TOP, (const void *)Reset_Handler
};
#endif
