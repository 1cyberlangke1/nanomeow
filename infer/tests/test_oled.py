"""OLED 显示层闸门：初始化序列、字形落点、换行、滚动、兜底 —— 全部按 SSD1306 的页规则核对。

输入：infer/c 的 nm_oled.c / nm_oled.h / nm_oled_selftest.c（自检里带一个假 I2C 总线）。
输出：pytest 断言；编译不过或自检失败就报错。
预期行为：显示行为由 C 自检里「重建显示 RAM」判，Python 侧只负责跑它，并守住两条与硬件
          无关的硬约束（提示词模板、显示层不许出现浮点）。
"""

import pathlib
import re
import shutil
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve()
C_DIR = HERE.parents[1] / "c"
# 头文件分散在 engine / generated / display / platform 四个子目录，编译时一起加 -I
C_INCLUDES = [a for p in ("", "engine", "generated", "display", "platform")
              for a in ("-I", str(C_DIR / p))]
# 板级入口住在 Keil 工程里（keil_demo/User/main.c），两条构建路径共用同一份，不再是 c/nm_fw.c。
FW_MAIN = HERE.parents[2] / "keil_demo" / "User" / "main.c"
GCC = shutil.which("gcc")


@pytest.mark.skipif(GCC is None, reason="本机没有 gcc")
def test_oled_selftest(tmp_path):
    """输入：tmp_path；输出：无。预期行为：C 自检编译无警告并通过。"""
    exe = tmp_path / "nm_oled_selftest.exe"
    subprocess.run([GCC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", *C_INCLUDES,
                    "-o", str(exe), str(C_DIR / "selftest" / "nm_oled_selftest.c"),
                    str(C_DIR / "display" / "nm_oled.c"), str(C_DIR / "generated" / "nm_font.c")],
                   check=True, capture_output=True, text=True)
    proc = subprocess.run([str(exe)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith("OK "), proc.stdout


def test_init_sequence_covers_128x64():
    """输入：无；输出：无。预期行为：初始化序列必须开电荷泵、置复用比 64、用水平寻址、开显示。"""
    src = (C_DIR / "display" / "nm_oled.c").read_text(encoding="utf-8")
    m = re.search(r"nm_init_seq\[\] = \{(.*?)\};", src, re.S)
    assert m, "nm_oled.c 里找不到初始化序列"
    seq = [int(v, 16) for v in re.findall(r"0x[0-9A-Fa-f]{2}", m.group(1))]
    assert seq[0] == 0xAE and seq[-1] == 0xAF, "必须「显示关」开头、「显示开」结尾"
    for pair in ([0x8D, 0x14], [0xA8, 0x3F], [0x20, 0x00], [0xDA, 0x12]):
        assert any(seq[i:i + 2] == pair for i in range(len(seq) - 1)), "缺少命令 %s" % pair


def test_firmware_prompt_template_has_no_space_after_colon():
    """输入：无；输出：无。预期行为：固件拼的提示词是 user:<内容>\\nbot:，冒号后没有空格。"""
    src = FW_MAIN.read_text(encoding="utf-8")
    assert '"user:"' in src and '"bot:"' in src
    assert '"user: ' not in src and '"bot: ' not in src


def test_oled_layer_is_float_free():
    """输入：无；输出：无。预期行为：显示层与上板入口不许出现 float / double。"""
    for path in (C_DIR / "display" / "nm_oled.c", C_DIR / "display" / "nm_oled_port.c", FW_MAIN):
        src = path.read_text(encoding="utf-8")
        assert not re.search(r"\b(float|double)\b", src), "%s 里出现了浮点类型" % path.name
