# 主机侧构建：演示 CLI（nm_chat）与四个自检程序。
# Cortex-M3 的尺寸账本与上板说明见 ../README.md。
cmake_minimum_required(VERSION 3.16)
project(nanomeow C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 与 infer/tests 里的编译口径保持一致
if(CMAKE_C_COMPILER_ID MATCHES "GNU")
  add_compile_options(-Wall -Wextra -Werror)
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

# 源码按功能分目录：engine 引擎本体、generated 代码生成产物、display OLED 显示层、
# platform 板级寄存器与引脚、host 主机侧程序、selftest 自检程序、tools 生成脚本。
# 头文件分散在多个子目录，统一加进搜索路径，源码里的 #include 不用带目录前缀。
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/engine
  ${CMAKE_CURRENT_SOURCE_DIR}/generated
  ${CMAKE_CURRENT_SOURCE_DIR}/display
  ${CMAKE_CURRENT_SOURCE_DIR}/platform
)

# 引擎本体：前向与 wkv7 递推、权重描述表、生成与停止、8x8 子集字库、OLED 显示层
# （显示层的传输钩子由调用方提供，上板是 display/nm_oled_port.c，主机端自检用假总线）
add_library(nanomeow STATIC
  engine/nanomeow.c
  engine/nm_gen.c
  generated/nm_weights.c
  generated/nm_font.c
  display/nm_oled.c
)

add_executable(nm_chat host/nm_chat.c)
target_link_libraries(nm_chat PRIVATE nanomeow)

# 主机吞吐基准：只量 nm_forward_token 的周期/token（主机口径，见文件头注释）
add_executable(nm_bench host/nm_bench.c)
target_link_libraries(nm_bench PRIVATE nanomeow)

add_executable(nm_engine_selftest selftest/nm_engine_selftest.c)
target_link_libraries(nm_engine_selftest PRIVATE nanomeow)

# 字库只依赖自己的常量表与 nm_utf8.h
add_executable(nm_font_selftest selftest/nm_font_selftest.c generated/nm_font.c)

# OLED 显示层：自检里自己实现 nm_oled_bus_write，所以只编 nm_oled.c 与字库
add_executable(nm_oled_selftest selftest/nm_oled_selftest.c display/nm_oled.c generated/nm_font.c)

# 这两个只依赖头文件里的定点内核与 UTF-8 解码器
add_executable(nm_fixed_selftest selftest/nm_fixed_selftest.c)
add_executable(nm_utf8_selftest selftest/nm_utf8_selftest.c)