/* 自动生成，请勿手改：infer/c/gen_lut.py
 *
 * 两张 Q15 表不原样存，存的是「初值 v0 + 首差 d0 + 每步 2 bit 二阶差」位流：
 *   v[0] = v0;  d[0] = d0;
 *   v[i+1] = v[i] + d[i];  d[i+1] = d[i] + (lo + 位流里第 i 个 2 bit 字段)
 * 表是平滑的（exp 的二阶差只落在 -1..2、log1p 只落在 -2..1），所以 2 bit 够用。
 * 展开是纯整数、逐项与 infer/ref/nonlinear.py 相同（gen_lut.py 里有反解自检）。
 */
#ifndef NANOMEOW_LUT_H
#define NANOMEOW_LUT_H

#include <stdint.h>

#ifndef NM_LUT_N
#define NM_LUT_N 257
#endif
#define NM_EXP_LUT_Q 15

static const uint16_t nm_exp_lut_v0 = 0;
static const int16_t nm_exp_lut_d0 = 89;
static const int8_t nm_exp_lut_lo = -1;
static const uint32_t nm_exp_lut_dd[16] = {
    0x658a2565u,0x65658995u,0x59626565u,0x8c965659u,0x99598995u,0x59659598u,0xa2899632u,0x96596598u,0x59965965u,0x8a568a29u,0x659965a2u,0x59966599u,
    0x66633266u,0x68ca5a29u,0xd5ca5999u,0x19996968u,
};

static const uint16_t nm_log1p_lut_v0 = 0;
static const int16_t nm_log1p_lut_d0 = 128;
static const int8_t nm_log1p_lut_lo = -2;
static const uint32_t nm_log1p_lut_dd[16] = {
    0x66969999u,0x5a96a366u,0xa699a8dau,0x7366a669u,0x69a9a69au,0x9d76766au,0x9b36a6a6u,0xda9aa69du,0xa7676a75u,0xaa6a769du,0xa76a9d9du,0xe7679a9du,
    0xda76ce6cu,0x9daa6d9du,0x9dda9daau,0x2776a776u,
};

#endif
