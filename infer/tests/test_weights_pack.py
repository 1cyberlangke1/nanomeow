"""无损压缩闸门：把 nm_weights.c 里的 scale 位流与行池反解回来，必须与 model_weights.h 逐位相同。

输入：infer/model_weights.h（原始 mul/shift/码）、infer/c/nm_weights.c（压缩后的位流与行池）。
输出：pytest 断言 —— 每个张量每一行的 (mul, shift) 与每个 int8 码都还原得一模一样。
预期行为：这条用例不需要 checkpoint，纯粹验证「编码 → 解码」是无损的；端到端的
          数值正确性由 test_c_engine.py 的 G1 闸门对拍独立的 Python 定点参考。
"""

import pathlib
import re

HERE = pathlib.Path(__file__).resolve()
C_DIR = HERE.parents[1] / "c"
HEADER = HERE.parents[1] / "model_weights.h"
GENERATED = C_DIR / "nm_weights.c"


def int_list(text, sym):
    """输入：文本、符号名；输出：该符号的整数列表（数组与标量都归一成 list），没有则 None。"""
    m = re.search(r"static const \w+ %s\[\] = \{([^{}]*)\};" % re.escape(sym), text)
    if m:
        return [int(v) for v in m.group(1).replace("\n", " ").split(",") if v.strip()]
    m = re.search(r"static const \w+ %s = (-?\d+);" % re.escape(sym), text)
    if m:
        return [int(m.group(1))]
    return None


def read_bits(blob, bitpos, n):
    """输入：字节串、起始位号、位宽；输出：小端位序的 n 位无符号值。预期行为：只读需要的字节。"""
    off = bitpos & 7
    q = bitpos >> 3
    acc = 0
    for i in range((n + off + 7) // 8):
        acc |= blob[q + i] << (8 * i)
    return (acc >> off) & ((1 << n) - 1)


def decode(blob, rows, cols, per_row, idx_bits, pool):
    """输入：scale 位流、行数、列数、per_row、索引位宽、行池；输出：(muls, shifts, codes)。

    预期行为：与 nanomeow.c 的 nm_row_scale / nm_row_codes 同一口径，反解出每一行的
              (mul, shift) 与每一行的 int8 码。
    """
    if not per_row:
        v = int.from_bytes(blob, "little")
        return [(v >> 7 | (1 << 23)) << 7], [(v & 0x7F) - 127], pool
    mb = (rows * 23 + 7) // 8
    base = blob[mb] - 256 if blob[mb] >= 128 else blob[mb]
    w = blob[mb + 1]
    db = (rows * w + 7) // 8 if w else 0
    muls, shifts, codes = [], [], []
    for r in range(rows):
        muls.append((read_bits(blob, r * 23, 23) | (1 << 23)) << 7)
        sh = base + (read_bits(blob, (mb + 2) * 8 + r * w, w) if w else 0)
        shifts.append(sh)
        idx = read_bits(blob, (mb + 2 + db) * 8 + r * idx_bits, idx_bits) if idx_bits else r
        codes.extend(pool[idx * cols:(idx + 1) * cols])
    return muls, shifts, codes


def test_scale_and_codes_roundtrip_is_lossless():
    """输入：无；输出：无。预期行为：105 个张量全部逐位还原，且没有漏掉任何一个描述符。"""
    header = HEADER.read_text(encoding="utf-8")
    gen = GENERATED.read_text(encoding="utf-8")
    blobs = {m.group(1): bytes(int(v) for v in m.group(2).split(",") if v.strip())
             for m in re.finditer(r"static const uint8_t nmwp_(\w+)\[\] = \{([^{}]*)\};", gen)}
    pools = {m.group(1): [int(v) for v in m.group(2).split(",") if v.strip()]
             for m in re.finditer(r"static const int8_t nmwc_(\w+)\[\] = \{([^{}]*)\};", gen)}
    mats = re.findall(r"\{\s*(\w+),\s*nmwp_(\w+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\}", gen)
    assert len(mats) == 105, "描述符个数不对：%d" % len(mats)

    checked = 0
    for codes_sym, name, rows, cols, per_row, idx_bits in mats:
        rows, cols, per_row, idx_bits = int(rows), int(cols), int(per_row), int(idx_bits)
        orig_codes = int_list(header, "nmw_" + name)
        orig_muls = int_list(header, "nmw_" + name + "_mul")
        orig_shifts = int_list(header, "nmw_" + name + "_shift")
        assert orig_codes is not None and orig_muls is not None and orig_shifts is not None, name
        assert len(orig_muls) == rows, "%s 的行数不一致" % name
        assert len(orig_codes) == rows * cols, "%s 的码数不一致" % name
        pool = pools.get(name, orig_codes)
        if idx_bits:
            assert codes_sym == "nmwc_" + name, "%s 用了索引却没指向行池" % name
            assert len(pool) % cols == 0 and len(pool) < len(orig_codes)
        else:
            assert codes_sym == "nmw_" + name, "%s 没去重却指向了行池" % name
        muls, shifts, codes = decode(blobs[name], rows, cols, per_row, idx_bits, pool)
        assert muls == orig_muls, "%s 的 mul 还原不一致" % name
        assert shifts == orig_shifts, "%s 的 shift 还原不一致" % name
        assert codes == orig_codes, "%s 的 int8 码还原不一致" % name
        checked += 1
    assert checked == 105


def test_generated_sizes_are_compressed():
    """输入：无；输出：无。预期行为：压缩后合计必须严格小于未压缩的 4 B/条 scale + 原始码。"""
    header = HEADER.read_text(encoding="utf-8")
    gen = GENERATED.read_text(encoding="utf-8")
    blob_total = sum(len(bytes(int(v) for v in m.group(1).split(",") if v.strip()))
                     for m in re.finditer(r"static const uint8_t nmwp_\w+\[\] = \{([^{}]*)\};", gen))
    pool_total = sum(len(m.group(1).split(","))
                     for m in re.finditer(r"static const int8_t nmwc_\w+\[\] = \{([^{}]*)\};", gen))
    codes_total = 0
    scale_entries = 0
    for m in re.finditer(r"static const \w+ (nmw_\w+)(\[\])? = \{([^{}]*)\};", header):
        name, is_arr, body = m.group(1), m.group(2), m.group(3)
        n = body.count(",") + 1
        if name.endswith("_mul") and is_arr:
            scale_entries += n
        elif not name.endswith("_mul") and not name.endswith("_shift"):
            codes_total += n
    assert blob_total + pool_total < codes_total + scale_entries * 4