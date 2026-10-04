/* nanomeow 上板固件入口：USART1 收一行 → 模型生成 → SSD1306 显示（同时把同一份字节回显到串口）。
 *
 * 输入：USART1（PA10，115200 8N1）收字节，'\n' 结束一行；输出：SSD1306 显示 + USART1（PA9）回显。
 * 预期行为：上电把系统时钟配到 72 MHz（HSE 8 MHz × 9），点屏并显示提示，然后循环
 *           「显示 user: 行 → 等一行输入 → 显示 user:<内容> / bot: → 跑模型 → 边生成边刷屏」。
 *           提示词严格按训练模板拼：user:<内容>\nbot:（冒号后没有空格）。全程无浮点。
 * 说明：这是上板用的最小入口（含复位向量与中断向量表），主机端对拍走 nm_chat。
 */
#include <string.h>

#include "nm_gen.h"
#include "nm_oled.h"
#include "nm_stm32f103.h"

#define NM_UART_BRR_115200_72M 0x271u  /* PCLK2 = 72 MHz 下的 115200：39.0625 → 尾数 39、小数 1 */
#define NM_LINE_MAX 256                /* 一行输入的上限（约 85 个汉字）；超出部分丢掉，不溢出 */
#define NM_PROMPT_MAX (NM_LINE_MAX + 16)

static nm_layer_state g_states[NM_N_LAYER];
static uint8_t g_line[NM_LINE_MAX];
static uint8_t g_prompt[NM_PROMPT_MAX];
/* 与主机演示同一口径：最多 128 个字节、重复惩罚关闭（65536 = 1.0）、窗口 0。 */
static const nm_gen_cfg g_cfg = { 128, 65536, 0 };

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

/* 输入：无；输出：无。预期行为：PA9 = 复用推挽输出、PA10 = 浮空输入，USART1 开 115200 收发。 */
static void nm_uart_init(void)
{
    NM_RCC_APB2ENR |= NM_APB2_IOPAEN | NM_APB2_USART1EN;
    NM_GPIOA_CRH = (NM_GPIOA_CRH & ~0x00000FF0u) | 0x000004B0u;   /* PA9 = 0xB、PA10 = 0x4 */
    NM_USART1_BRR = NM_UART_BRR_115200_72M;
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

/* 输入：上下文（未用）、一个完整字符的字节与长度；输出：无。
 * 预期行为：同一份字节既回显到串口，也画到屏上并只刷脏页 —— 生成过程中就能看到字出来。 */
static void nm_fw_emit(void *ctx, const uint8_t *bytes, int n)
{
    int i;

    (void)ctx;
    for (i = 0; i < n; i++) nm_uart_putc(bytes[i]);
    nm_oled_put_utf8(bytes, n);
    nm_oled_flush();
}

int main(void)
{
    nm_sys_clock_72m();
    nm_uart_init();
    nm_oled_port_init();
    nm_delay_ms(100);              /* SSD1306 上电后要等电源稳定才收命令 */
    nm_oled_init();
    nm_oled_puts((const uint8_t *)"nanomeow ready", 14);
    nm_oled_flush();

    for (;;) {
        int n, p = 0, hit = 0;

        nm_oled_clear();
        nm_oled_puts((const uint8_t *)"user:", 5);
        nm_oled_flush();

        n = nm_uart_read_line(g_line, NM_LINE_MAX);
        if (n <= 0) continue;

        nm_oled_puts(g_line, n);
        nm_oled_flush();

        memcpy(g_prompt + p, "user:", 5); p += 5;
        memcpy(g_prompt + p, g_line, (size_t)n); p += n;
        g_prompt[p++] = '\n';
        memcpy(g_prompt + p, "bot:", 4); p += 4;

        nm_oled_newline();
        nm_oled_puts((const uint8_t *)"bot:", 4);
        nm_oled_flush();

        nm_generate(g_states, g_prompt, p, &g_cfg, nm_fw_emit, 0, &hit);
        nm_uart_putc('\n');
        nm_oled_newline();
        nm_oled_flush();
    }
}

/* 上板入口：复位后直接进 main，跑完不回。中断向量表只有栈顶与复位两个表项（本固件不开中断）。 */
void Reset_Handler(void)
{
    (void)main();
    for (;;) { }
}

static const int32_t g_stack_top = 0x20005000;

__attribute__((section(".isr_vector"), used))
static const void *g_vectors[48] = {
    (const void *)&g_stack_top, (const void *)Reset_Handler
};