"""无损压缩闸门：把 nm_weights.c 里的权重池反解回来，必须与 model_weights.h 逐位相同。

输入：infer/model_weights.h（原始 mul/shift/码）、infer/c/nm_weights.c（压缩后的权重池与描述符）。
输出：pytest 断言 —— 每个张量每一行的 (mul, shift) 与每个 int8 码都还原得一模一样。
预期行为：这条用例不需要 checkpoint，纯粹验证「编码 → 解码」是无损的；端到端的
          数值正确性由 test_c_engine.py 的 G1 闸门对拍独立的 Python 定点参考。
          描述符里的 code_off / scale_off 是统一权重池 nm_pool 的 uint16 字节偏移，
          同一个张量的码与 scale 位流在池里首尾相接，所以码长 = scale_off - code_off。
"""

import pathlib
import re

HERE = pathlib.Path(__file__).resolve()
C_DIR = HERE.parents[1] / "c"
HEADER = HERE.parents[1] / "model_weights.h"
GENERATED = C_DIR / "nm_weights.c"

# 带 /* nmw_xxx */ 尾注释的描述符才是真实张量；层 1/2 的 ln0 是空占位，没有注释。
DESC_RE = re.compile(
    r"\{\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\}"
    r"[\s,;]*/\*\s*(nmw_\w+)\s*\*/")
POOL_RE = re.compile(r"const uint8_t nm_pool\[(\d+)\] = \{([^{}]*)\};")


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


def scale_blob_len(blob, rows, per_row, idx_bits):
    """输入：scale 位流、行数、per_row、索引位宽；输出：该位流占用的字节数。

    预期行为：与 gen_weights.py 的打包布局一致 —— per-tensor 恒 4 B；per-row 是
              [尾数 23 位][1 B 基线][1 B 增量位宽][增量][索引]，后两段按需取整。
    """
    if not per_row:
        return 4
    mb = (rows * 23 + 7) // 8
    w = blob[mb + 1]
    db = (rows * w + 7) // 8 if w else 0
    ib = (rows * idx_bits + 7) // 8 if idx_bits else 0
    return mb + 2 + db + ib


def decode(blob, rows, cols, per_row, idx_bits, codes):
    """输入：scale 位流、行数、列数、per_row、索引位宽、行池码；输出：(muls, shifts, 全部码)。

    预期行为：与 nanomeow.c 的 nm_row_scale / nm_row_codes 同一口径，反解出每一行的
              (mul, shift) 与每一行的 int8 码。
    """
    if not per_row:
        v = int.from_bytes(blob, "little")
        return [(v >> 7 | (1 << 23)) << 7], [(v & 0x7F) - 127], codes
    mb = (rows * 23 + 7) // 8
    base = blob[mb] - 256 if blob[mb] >= 128 else blob[mb]
    w = blob[mb + 1]
    db = (rows * w + 7) // 8 if w else 0
    muls, shifts, out = [], [], []
    for r in range(rows):
        muls.append((read_bits(blob, r * 23, 23) | (1 << 23)) << 7)
        sh = base + (read_bits(blob, (mb + 2) * 8 + r * w, w) if w else 0)
        shifts.append(sh)
        idx = read_bits(blob, (mb + 2 + db) * 8 + r * idx_bits, idx_bits) if idx_bits else r
        out.extend(codes[idx * cols:(idx + 1) * cols])
    return muls, shifts, out


def parse_generated():
    """输入：无；输出：(池字节串, [(code_off, scale_off, rows, cols, per_row, idx_bits, 符号)])。

    预期行为：池声明长度必须等于实际元素个数，否则直接报错，避免静默错位。
    """
    gen = GENERATED.read_text(encoding="utf-8")
    m = POOL_RE.search(gen)
    assert m, "nm_weights.c 里没有找到 nm_pool"
    declared, body = int(m.group(1)), bytes(int(v) for v in m.group(2).split(",") if v.strip())
    assert len(body) == declared, "nm_pool 声明 %d B、实际 %d B" % (declared, len(body))
    descs = [tuple(int(g) for g in mm.groups()[:6]) + (mm.group(7),)
             for mm in DESC_RE.finditer(gen)]
    return body, descs


def test_scale_and_codes_roundtrip_is_lossless():
    """输入：无；输出：无。预期行为：105 个张量全部逐位还原，且没有漏掉任何一个描述符。"""
    header = HEADER.read_text(encoding="utf-8")
    pool, descs = parse_generated()
    assert len(descs) == 105, "描述符个数不对：%d" % len(descs)
    assert len({d[6] for d in descs}) == 105, "描述符符号有重复"

    for code_off, scale_off, rows, cols, per_row, idx_bits, sym in descs:
        assert code_off <= scale_off <= len(pool), "%s 的偏移越界" % sym
        # 池是 uint8_t，权重码按 int8 存，先补符号位再反解
        codes = [b - 256 if b >= 128 else b for b in pool[code_off:scale_off]]
        blob = pool[scale_off:scale_off + scale_blob_len(pool[scale_off:], rows, per_row, idx_bits)]
        orig_codes = int_list(header, sym)
        orig_muls = int_list(header, sym + "_mul")
        orig_shifts = int_list(header, sym + "_shift")
        assert orig_codes is not None and orig_muls is not None and orig_shifts is not None, sym
        assert len(orig_muls) == rows, "%s 的行数不一致" % sym
        assert len(orig_codes) == rows * cols, "%s 的码数不一致" % sym
        assert len(codes) % cols == 0, "%s 的行池不是整行" % sym
        if idx_bits:
            assert len(codes) < rows * cols, "%s 用了索引却没有省行" % sym
        muls, shifts, codes = decode(blob, rows, cols, per_row, idx_bits, codes)
        assert muls == orig_muls, "%s 的 mul 还原不一致" % sym
        assert shifts == orig_shifts, "%s 的 shift 还原不一致" % sym
        assert codes == orig_codes, "%s 的 int8 码还原不一致" % sym


def test_generated_sizes_are_compressed():
    """输入：无；输出：无。预期行为：权重池必须严格小于未压缩的 4 B/条 scale + 原始码。"""
    header = HEADER.read_text(encoding="utf-8")
    pool, _ = parse_generated()
    codes_total = 0
    scale_entries = 0
    for m in re.finditer(r"static const \w+ (nmw_\w+)(\[\])? = \{([^{}]*)\};", header):
        name, is_arr, body = m.group(1), m.group(2), m.group(3)
        n = body.count(",") + 1
        if name.endswith("_mul") and is_arr:
            scale_entries += n
        elif not name.endswith("_mul") and not name.endswith("_shift"):
            codes_total += n
    assert len(pool) < codes_total + scale_entries * 4
