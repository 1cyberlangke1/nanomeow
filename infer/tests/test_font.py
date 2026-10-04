"""字库闸门：生成的 C 字库 == 子集清单 == BDF 原件，逐位一致。

输入：infer/font/subset_8x8.txt、infer/c/generated/nm_font.c、infer/c/generated/nm_font.h，
      以及（本机有的话）tmp/f8bdf 下的 fusion-pixel 8px 等宽 BDF。
输出：pytest 断言；BDF 不在时那条用例 skip（干净 clone 没有 tmp/）。
预期行为：C 里的码点表 / 点阵位流必须与清单逐字节相同；清单里的每个字形必须与 BDF 原件
          按基线摆格后逐位相同 —— 这条链断了，OLED 上就会出现错字。
"""

import pathlib
import re
import shutil
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "infer"))

from font.select_subset import parse_bdf, packed, to_cell  # noqa: E402

C_DIR = REPO / "infer" / "c"
# 头文件分散在 engine / generated / display / platform 四个子目录，编译时一起加 -I
C_INCLUDES = [a for p in ("", "engine", "generated", "display", "platform")
              for a in ("-I", str(C_DIR / p))]
SUBSET = REPO / "infer" / "font" / "subset_8x8.txt"
BDF_DIR = REPO / "tmp" / "f8bdf"
GCC = shutil.which("gcc")

ROW_BYTES = 7     # 每字存的行数（原字体第 0 行恒空）
INK_BITS = 7      # 每行存的列数（原字体第 7 列恒空）


def read_subset():
    """输入：无；输出：[(码点, 7 字节点阵)]。预期行为：忽略 `#` 注释行。"""
    out = []
    for line in SUBSET.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        cp_hex, blob_hex, _ = line.split(" ", 2)   # 第 3 段是注释字符，可能本身就是空格
        out.append((int(cp_hex, 16), bytes.fromhex(blob_hex)))
    return out


def read_c_array(name):
    """输入：C 数组名；输出：它的 int 值列表。预期行为：找不到数组就断言失败。"""
    text = (C_DIR / "generated" / "nm_font.c").read_text(encoding="utf-8")
    m = re.search(r"const uint8_t %s\[(\d+)\] = \{(.*?)\};" % re.escape(name), text, re.S)
    assert m, "nm_font.c 里找不到数组 %s" % name
    values = [int(v) for v in m.group(2).replace("\n", " ").split(",") if v.strip()]
    assert len(values) == int(m.group(1)), "%s 声明长度与实际项数不一致" % name
    return values


def decode_cp(values):
    """输入：码点表字节；输出：码点列表。预期行为：LEB128 增量流必须正好用完整张表。"""
    cps, pos, prev = [], 0, 0
    while pos < len(values):
        value, shift = 0, 0
        while True:
            assert pos < len(values), "码点表提前用尽"
            byte = values[pos]
            pos += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                break
        prev += value
        cps.append(prev)
    return cps


def unpack(packed, n):
    """输入：点阵位流、字数；输出：[7 字节点阵]。

    预期行为：与 gen_font.py 的 pack_glyphs 互逆 —— 每字 NM_FONT_ROW*NM_FONT_INK 位，
              第 r 行 7 位、MSB 在前；第 k 位对应点阵的 bit(7-k)，解出的值再 <<1 放回 bit7..bit1。
    """
    out = []
    for i in range(n):
        blob = bytearray()
        for r in range(ROW_BYTES):
            value = 0
            for k in range(INK_BITS):
                bit = i * ROW_BYTES * INK_BITS + r * INK_BITS + k
                value = (value << 1) | ((packed[bit >> 3] >> (7 - (bit & 7))) & 1)
            blob.append((value << 1) & 0xFF)
        out.append(bytes(blob))
    return out


def test_header_matches_c():
    """输入：无；输出：无。预期行为：nm_font.h 的常量与 nm_font.c 的实际长度一致。"""
    header = (C_DIR / "generated" / "nm_font.h").read_text(encoding="utf-8")
    consts = {k: int(v) for k, v in re.findall(r"#define (NM_FONT_\w+) (\d+)", header)}
    assert consts["NM_FONT_N"] == len(read_subset())
    assert consts["NM_FONT_CP_BYTES"] == len(read_c_array("nm_font_cp"))
    assert consts["NM_FONT_PACKED_BYTES"] == len(read_c_array("nm_font_packed"))
    assert consts["NM_FONT_ROW"] == ROW_BYTES and consts["NM_FONT_INK"] == INK_BITS
    assert consts["NM_FONT_ROW"] == 7 and consts["NM_FONT_W"] == 8 and consts["NM_FONT_H"] == 8


def test_c_matches_subset():
    """输入：无；输出：无。预期行为：C 里的码点与点阵必须与清单逐字节相同。"""
    subset = read_subset()
    assert decode_cp(read_c_array("nm_font_cp")) == [cp for cp, _ in subset]
    packed = read_c_array("nm_font_packed")
    assert unpack(packed, len(subset)) == [blob for _, blob in subset]
    assert len(read_c_array("nm_font_fallback")) == 7


def test_subset_is_compact():
    """输入：无；输出：无。预期行为：打包后的点阵 + 码点表不超预算，且打包后确实比原样小。"""
    header = SUBSET.read_text(encoding="utf-8")
    budget = int(re.search(r"字节上限 (\d+)", header).group(1))
    packed = len(read_c_array("nm_font_packed")) + len(read_c_array("nm_font_cp"))
    assert packed <= budget, "字库 %d B 超预算 %d B" % (packed, budget)
    raw = len(read_subset()) * ROW_BYTES + len(read_c_array("nm_font_cp"))
    assert packed < raw, "打包后 %d B 没比原 %d B 小" % (packed, raw)


@pytest.mark.skipif(not BDF_DIR.exists(), reason="本机没有 tmp/f8bdf（干净 clone 不带走 BDF）")
def test_subset_matches_bdf():
    """输入：无；输出：无。预期行为：清单里每个字形与 BDF 原件按基线摆格后逐位相同。"""
    ascent, glyphs = None, {}
    for name in ("fusion-pixel-8px-monospaced-latin.bdf",
                 "fusion-pixel-8px-monospaced-zh_hans.bdf"):
        asc, g = parse_bdf(BDF_DIR / name)
        ascent = asc if ascent is None else ascent
        glyphs.update(g)
    for cp, blob in read_subset():
        assert cp in glyphs, "BDF 里没有 U+%04X" % cp
        assert packed(to_cell(*glyphs[cp], ascent)) == blob, "U+%04X 的点阵与 BDF 不一致" % cp


@pytest.mark.skipif(GCC is None, reason="本机没有 gcc")
def test_c_selftest(tmp_path):
    """输入：tmp_path；输出：无。预期行为：C 自检编译无警告并通过。"""
    exe = tmp_path / "nm_font_selftest.exe"
    subprocess.run([GCC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", *C_INCLUDES,
                    "-o", str(exe), str(C_DIR / "selftest" / "nm_font_selftest.c"),
                    str(C_DIR / "generated" / "nm_font.c")],
                   check=True, capture_output=True, text=True)
    proc = subprocess.run([str(exe)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith("OK "), proc.stdout
