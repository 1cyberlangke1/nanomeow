/* 交叉编译到 armv7m-none-eabi 时的最小 string.h：只声明引擎真正用到的四个函数。
 * 实现由链接期的 C 运行库提供，这里只补 freestanding 下缺失的声明。 */
#ifndef NANOMEOW_ARM_STRING_H
#define NANOMEOW_ARM_STRING_H
#include <stddef.h>
void *memcpy(void *dst, const void *src, size_t n);
void *memmove(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);
int memcmp(const void *a, const void *b, size_t n);
#endif