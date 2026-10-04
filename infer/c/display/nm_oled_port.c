/* nanomeow 的 SSD1306 传输层：STM32F103C8T6 上软件 I2C 的位操作。
 *
 * 为什么软件 I2C：SSD1306 只要 ~400 kHz，位操作够用，还省掉硬件 I2C 的事件状态机与中断
 * （那套在 -Oz 下也要几百字节）。两个引脚都配成开漏输出，SDA 靠读 IDR 取回总线电平收 ACK。
 * 半周期长度由 NM_OLED_I2C_DELAY 调：72 MHz 下大约 (count * 4 + 8) 个周期，16 约合 1 µs。
 * SCL / SDA 的引脚号来自 nm_board.h（工程里由 User/config.h 指定），两个脚必须同属 GPIOB。
 */
#include "nm_oled.h"
#include "nm_board.h"
#include "nm_stm32f103.h"

#define NM_OLED_SCL_BIT (1u << NM_OLED_SCL_PIN)
#define NM_OLED_SDA_BIT (1u << NM_OLED_SDA_PIN)
#define NM_OLED_I2C_DELAY 16

static void nm_i2c_delay(void)
{
    for (volatile int i = 0; i < NM_OLED_I2C_DELAY; i++) { }
}

static void nm_i2c_scl_hi(void) { NM_GPIOB_BSRR = NM_OLED_SCL_BIT; }
static void nm_i2c_scl_lo(void) { NM_GPIOB_BRR = NM_OLED_SCL_BIT; }
static void nm_i2c_sda_hi(void) { NM_GPIOB_BSRR = NM_OLED_SDA_BIT; }
static void nm_i2c_sda_lo(void) { NM_GPIOB_BRR = NM_OLED_SDA_BIT; }
static int nm_i2c_sda_read(void) { return (int)((NM_GPIOB_IDR >> NM_OLED_SDA_PIN) & 1u); }

/* 输入：无；输出：无。预期行为：SCL 高时 SDA 由高变低 = 起始条件。 */
static void nm_i2c_start(void)
{
    nm_i2c_sda_hi(); nm_i2c_scl_hi(); nm_i2c_delay();
    nm_i2c_sda_lo(); nm_i2c_delay();
    nm_i2c_scl_lo(); nm_i2c_delay();
}

/* 输入：无；输出：无。预期行为：SCL 高时 SDA 由低变高 = 停止条件。 */
static void nm_i2c_stop(void)
{
    nm_i2c_sda_lo(); nm_i2c_delay();
    nm_i2c_scl_hi(); nm_i2c_delay();
    nm_i2c_sda_hi(); nm_i2c_delay();
}

/* 输入：一个字节；输出：1 = 从机给了 ACK，0 = 没给。预期行为：MSB 先发，第 9 个时钟读 ACK。 */
static int nm_i2c_put(uint8_t byte)
{
    int i, ack;

    for (i = 7; i >= 0; i--) {
        if (byte & (uint8_t)(1u << i)) nm_i2c_sda_hi(); else nm_i2c_sda_lo();
        nm_i2c_delay();
        nm_i2c_scl_hi(); nm_i2c_delay();
        nm_i2c_scl_lo(); nm_i2c_delay();
    }
    nm_i2c_sda_hi(); nm_i2c_delay();       /* 释放 SDA，让从机有机会拉低 */
    nm_i2c_scl_hi(); nm_i2c_delay();
    ack = !nm_i2c_sda_read();
    nm_i2c_scl_lo(); nm_i2c_delay();
    return ack;
}

/* 输入：引脚号 0..15；输出：无。预期行为：把该引脚配成通用开漏输出（软件 I2C 的空闲态）。
 * 为什么不用 CMSIS：本工程只用 GPIOB 这一组寄存器，CRL 管 0..7、CRH 管 8..15，
 * 每脚 4 位（MODE = 11 输出 50 MHz、CNF = 01 通用开漏 = 0x7），引脚号自己算更省 Flash。 */
static void nm_gpio_od_out(int pin)
{
    volatile uint32_t *cr = pin < 8 ? &NM_GPIOB_CRL : &NM_GPIOB_CRH;
    int sh = (pin & 7) * 4;

    *cr = (*cr & ~(0xFu << sh)) | (0x7u << sh);
}

void nm_oled_port_init(void)
{
    NM_RCC_APB2ENR |= NM_APB2_IOPBEN;
    nm_gpio_od_out(NM_OLED_SCL_PIN);
    nm_gpio_od_out(NM_OLED_SDA_PIN);
    nm_i2c_sda_hi();
    nm_i2c_scl_hi();
}

void nm_oled_bus_write(uint8_t ctrl, const uint8_t *p, int n)
{
    int i, ok;

    nm_i2c_start();
    ok = nm_i2c_put((uint8_t)NM_OLED_I2C_ADDR);
    if (ok) ok = nm_i2c_put(ctrl);
    for (i = 0; ok && i < n; i++)
        ok = nm_i2c_put(p[i]);
    nm_i2c_stop();
}
