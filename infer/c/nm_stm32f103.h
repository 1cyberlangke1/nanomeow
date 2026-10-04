/* nanomeow 上板用到的 STM32F103 最小寄存器表（地址取自 RM0008 的存储器映射）。
 *
 * 为什么不引 CMSIS / HAL：上板工程多半自带一套，重名会冲突；这里只用 RCC / FLASH / GPIOA /
 * GPIOB / USART1 五个外设，手写地址比拖进整包头文件省 Flash，也不引入任何浮点依赖。
 */
#ifndef NANOMEOW_STM32F103_H
#define NANOMEOW_STM32F103_H

#include <stdint.h>

#define NM_RCC_CR      (*(volatile uint32_t *)0x40021000u)
#define NM_RCC_CFGR    (*(volatile uint32_t *)0x40021004u)
#define NM_RCC_APB2ENR (*(volatile uint32_t *)0x40021018u)
#define NM_FLASH_ACR   (*(volatile uint32_t *)0x40022000u)

#define NM_GPIOA_CRH   (*(volatile uint32_t *)0x40010804u)
#define NM_GPIOB_CRL   (*(volatile uint32_t *)0x40010C00u)
#define NM_GPIOB_IDR   (*(volatile uint32_t *)0x40010C08u)
#define NM_GPIOB_BSRR  (*(volatile uint32_t *)0x40010C10u)
#define NM_GPIOB_BRR   (*(volatile uint32_t *)0x40010C14u)

#define NM_USART1_SR   (*(volatile uint32_t *)0x40013800u)
#define NM_USART1_DR   (*(volatile uint32_t *)0x40013804u)
#define NM_USART1_BRR  (*(volatile uint32_t *)0x40013808u)
#define NM_USART1_CR1  (*(volatile uint32_t *)0x4001380Cu)

/* RCC_APB2ENR 的使能位 */
#define NM_APB2_IOPAEN   (1u << 2)
#define NM_APB2_IOPBEN   (1u << 3)
#define NM_APB2_USART1EN (1u << 14)

/* 输入：无；输出：无。预期行为：开 GPIOB 时钟，把 PB6 / PB7 配成通用开漏输出（软件 I2C 的空闲态）。
 * 由 nm_oled_port.c 实现；主机端不需要这个符号。 */
void nm_oled_port_init(void);

#endif