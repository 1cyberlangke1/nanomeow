/* STM32F103C8T6 的存储布局：64 KiB Flash / 20 KiB RAM。
 *
 * .ARM.exidx / .ARM.extab 是 ARM EABI 的栈展开索引，只有异常处理或调试器回溯才会用到。
 * 这个固件是裸机、无异常、无回溯需求的，所以整段丢弃 —— 实测省 512 B Flash，丢掉的
 * 唯一代价是调试器不能再沿着调用栈回溯。
 */
ENTRY(Reset_Handler)

MEMORY
{
  FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 64K
  RAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 20K
}

SECTIONS
{
  .isr_vector : { KEEP(*(.isr_vector)) } > FLASH
  .text       : { *(.text*) } > FLASH
  .rodata     : { *(.rodata*) } > FLASH
  .data       : { *(.data*) } > RAM AT> FLASH
  .bss        : { *(.bss*) *(COMMON) } > RAM

  /DISCARD/ : { *(.ARM.exidx*) *(.ARM.extab*) }
}