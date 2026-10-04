# 主机侧构建：演示 CLI（nm_chat）与三个自检程序。
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

# 引擎本体：前向与 wkv7 递推、权重描述表、生成与停止、8x8 子集字库
add_library(nanomeow STATIC nanomeow.c nm_gen.c nm_weights.c nm_font.c)
target_include_directories(nanomeow PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(nm_chat nm_chat.c)
target_link_libraries(nm_chat PRIVATE nanomeow)

add_executable(nm_engine_selftest nm_engine_selftest.c)
target_link_libraries(nm_engine_selftest PRIVATE nanomeow)

# 字库只依赖自己的常量表与 nm_utf8.h
add_executable(nm_font_selftest nm_font_selftest.c nm_font.c)

# 这两个只依赖头文件里的定点内核与 UTF-8 解码器
add_executable(nm_fixed_selftest nm_fixed_selftest.c)
add_executable(nm_utf8_selftest nm_utf8_selftest.c)
