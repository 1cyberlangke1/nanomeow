"""从 infer/font/subset_8x8.txt 生成 C 引擎的字库 infer/c/nm_font.c 与 nm_font.h。

输入：infer/font/subset_8x8.txt（每行「码点 7 行点阵 字符」，由 infer/font/select_subset.py 选出来）。
输出：infer/c/nm_font.h（常量与查表接口）与 infer/c/nm_font.c（码点表 + 打包点阵 + 兜底字形 + 查表实现）。
预期行为：码点按升序存成「首个绝对 + 之后增量」的 LEB128 流，顺序扫描即可二分退化成线性；
          点阵按「每字 NM_FONT_ROW * NM_FONT_INK = 49 位」紧密打包成位流 —— 每行只取 bit7..bit1
          （第 7 列恒空，不存）7 位、MSB 在前依次写入；表尾补 1 个 0 字节，因为 C 侧解包固定读
          「当前字节 + 下一个字节」，没有这个补位读最后一个字会越界一字节；
          文件头的注释由实际数据算出来（字数、点阵字节、码点表字节、合计），不写死；
          生成后把码点表和点阵位流都解回来与内存里的数据逐位比对，不一致就报错退出。
"""

import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SUBSET = HERE.parent / "font" / "subset_8x8.txt"
OUT_C = HERE / "nm_font.c"
OUT_H = HERE / "nm_font.h"

CELL = 8          # 8x8 点阵
ROW_BYTES = 7     # 只存第 1..7 行：原字体第 0 行恒空
INK_BITS = 7      # 行内只用 bit7..bit1：原字体第 7 列恒空
GLYPH_BITS = ROW_BYTES * INK_BITS   # 每字 49 位

# 兜底字形：7x7 空心方框，字库里没有的码点用它，避免 OLED 上出现空白或乱点
FALLBACK = (0x7C, 0x44, 0x44, 0x44, 0x44, 0x44, 0x7C)


def read_subset(path):
    """读子集清单。

    输入：subset_8x8.txt 路径。
    输出：[(码点, 7 字节点阵)]，按码点升序。
    预期行为：`#` 开头是注释；每行三段，第三段是原字符（只为方便人眼复核，解析时忽略）；
              点阵必须是 ROW_BYTES 字节，码点必须严格升序且不重复。
    """
    glyphs = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        parts = line.split(" ")
        if len(parts) != 3:
            raise SystemExit("自检失败：%s:%d 不是三段" % (path.name, lineno))
        cp = int(parts[0], 16)
        blob = bytes.fromhex(parts[1])
        if len(blob) != ROW_BYTES:
            raise SystemExit("自检失败：%s:%d 点阵不是 %d 字节" % (path.name, lineno, ROW_BYTES))
        if any(v & 1 for v in blob):
            raise SystemExit("自检失败：%s:%d 第 7 列非空" % (path.name, lineno))
        glyphs.append((cp, blob))
    if not glyphs:
        raise SystemExit("自检失败：%s 里一个字都没有" % path.name)
    cps = [cp for cp, _ in glyphs]
    if cps != sorted(set(cps)):
        raise SystemExit("自检失败：%s 的码点没有严格升序 / 有重复" % path.name)
    return glyphs


def leb128(value):
    """输入：非负整数；输出：它的 LEB128 字节列表。预期行为：低位在前，最高位是续接标志。"""
    out = []
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | 0x80 if value else byte)
        if not value:
            return out


def emit_array(fh, name, values, per_line=16):
    """输入：文件对象、数组名、值列表、每行个数；输出：无。

    预期行为：写成 `const uint8_t` 数组；不写 static，因为 nm_font.h 里是 extern 声明，
    自检程序要直接读这几张表。
    """
    fh.write("const uint8_t %s[%d] = {\n" % (name, len(values)))
    for i in range(0, len(values), per_line):
        fh.write("    " + ",".join(str(v) for v in values[i:i + per_line]) + ",\n")
    fh.write("};\n")


def build_cp_table(glyphs):
    """输入：升序字形表；输出：码点增量 LEB128 字节列表。预期行为：首个是绝对码点，之后是差。"""
    out, prev = [], 0
    for cp, _ in glyphs:
        out += leb128(cp - prev)
        prev = cp
    return out


def pack_glyphs(glyphs):
    """输入：升序字形表；输出：点阵位流 bytes（含表尾 1 个 0 补位）。

    预期行为：每字 GLYPH_BITS 位，第 r 行取 (byte >> 1) & 0x7F 的 7 位、MSB 在前依次写入；
              总位数向上取整到字节后再补 1 个 0 字节，供 C 侧固定读两字节的哨兵。
    """
    out = bytearray((len(glyphs) * GLYPH_BITS + 7) // 8 + 1)
    for i, (_, blob) in enumerate(glyphs):
        for r, byte in enumerate(blob):
            value = (byte >> 1) & 0x7F
            for k in range(INK_BITS):
                if value & (0x40 >> k):
                    bit = i * GLYPH_BITS + r * INK_BITS + k
                    out[bit >> 3] |= 0x80 >> (bit & 7)
    return bytes(out)


def unpack_glyphs(packed, n):
    """输入：点阵位流、字数；输出：[7 字节点阵]。预期行为：pack_glyphs 的逆运算，逐位取。"""
    glyphs = []
    for i in range(n):
        blob = bytearray()
        for r in range(ROW_BYTES):
            value = 0
            for k in range(INK_BITS):
                bit = i * GLYPH_BITS + r * INK_BITS + k
                value = (value << 1) | ((packed[bit >> 3] >> (7 - (bit & 7))) & 1)
            blob.append((value << 1) & 0xFF)
        glyphs.append(bytes(blob))
    return glyphs


def parse_cp_back(cp_bytes, n):
    """输入：码点表字节、字数；输出：码点列表。预期行为：LEB128 流必须恰好用完整张表。"""
    cps, pos, prev = [], 0, 0
    for _ in range(n):
        value, shift = 0, 0
        while True:
            if pos >= len(cp_bytes):
                raise SystemExit("自检失败：码点表提前用尽")
            byte = cp_bytes[pos]
            pos += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                break
        prev += value
        cps.append(prev)
    if pos != len(cp_bytes):
        raise SystemExit("自检失败：码点表有 %d 字节没用上" % (len(cp_bytes) - pos))
    return cps


def write_header(n, cp_bytes, packed):
    """输入：字数、码点表、点阵位流；输出：无。预期行为：常量全部由实际数据算出来。"""
    with OUT_H.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("/* 自动生成，请勿手改：infer/c/gen_font.py */\n")
        fh.write("#ifndef NANOMEOW_FONT_H\n#define NANOMEOW_FONT_H\n\n#include <stdint.h>\n\n")
        fh.write("#define NM_FONT_N %d          /* 收录字符数 */\n" % n)
        fh.write("#define NM_FONT_CP_BYTES %d   /* 码点表字节数 */\n" % len(cp_bytes))
        fh.write("#define NM_FONT_PACKED_BYTES %d   /* 点阵位流字节数（含表尾 1 个 0 补位） */\n"
                 % len(packed))
        fh.write("#define NM_FONT_W %d           /* 点阵宽（含原字体的右侧字间距列） */\n" % CELL)
        fh.write("#define NM_FONT_H %d           /* 点阵高（含原字体的顶部空行） */\n" % CELL)
        fh.write("#define NM_FONT_ROW %d         /* 实际存的像素行数：原字体第 0 行恒空 */\n" % ROW_BYTES)
        fh.write("#define NM_FONT_INK %d         /* 实际存的像素列数：原字体第 7 列恒空 */\n" % INK_BITS)
        fh.write("#define NM_FONT_GLYPH_BITS (NM_FONT_ROW * NM_FONT_INK)   /* 每字占的位数 */\n\n")
        fh.write("/* 码点表：LEB128 增量流，首个是绝对码点，之后是与前一项的差，按码点升序。 */\n")
        fh.write("extern const uint8_t nm_font_cp[NM_FONT_CP_BYTES];\n")
        fh.write("/* 点阵位流：NM_FONT_N 个字，每个 NM_FONT_GLYPH_BITS 位；第 r 行 7 位（原字节 bit7..bit1）\n"
                 " * MSB 在前。表尾多 1 个 0 字节，是解包时「固定读两个字节」的哨兵。 */\n")
        fh.write("extern const uint8_t nm_font_packed[NM_FONT_PACKED_BYTES];\n")
        fh.write("/* 兜底字形：7x7 空心方框，字库里没有的码点用它。 */\n")
        fh.write("extern const uint8_t nm_font_fallback[NM_FONT_ROW];\n\n")
        fh.write("""/* 输入：BMP 码点、至少 NM_FONT_ROW 字节的输出缓冲；输出：无。
 * 预期行为：把该字形解包进 out（每行 bit7..bit1 = 第 0..6 列，bit0 恒 0），码点不在字库时写兜底；
 *           顺序扫描码点表（升序），超过目标码点立刻退出；只读常量，可重入。 */
void nm_font_glyph(uint32_t cp, uint8_t *out);

/* 输入：一个完整字符的 UTF-8 字节、字节数、输出缓冲；输出：无。
 * 预期行为：同 nm_font_glyph；长度不够或序列非法时写兜底字形，不读越界。 */
void nm_font_glyph_utf8(const uint8_t *s, int n, uint8_t *out);

#endif
""")


def write_source(n, cp_bytes, packed):
    """输入：字数、码点表、点阵位流；输出：无。预期行为：C 查表实现与 Python 打包口径互逆。"""
    with OUT_C.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("/* 自动生成，请勿手改：infer/c/gen_font.py\n")
        fh.write(" * nanomeow 的 8x8 子集字库：%d 字，点阵 %d B（打包） + 码点表 %d B = %d B。\n"
                 % (n, len(packed), len(cp_bytes), len(packed) + len(cp_bytes)))
        fh.write(" * 字形来源：fusion-pixel-font 8px 等宽（OFL-1.1）；选字清单与来源说明见\n")
        fh.write(" * infer/font/subset_8x8.txt 与 infer/font/README.md。\n")
        fh.write(" * 布局：每字只存第 1..7 行（原字体第 0 行恒空），行内只用 bit7..bit1（第 7 列恒空）；\n")
        fh.write(" *       49 位一组紧密打包，不是字节对齐，解包按位取（见 nm_font_read7）。\n")
        fh.write(" */\n")
        fh.write("#include <stddef.h>\n\n#include \"nm_font.h\"\n#include \"nm_utf8.h\"\n\n")
        emit_array(fh, "nm_font_cp", cp_bytes)
        fh.write("\n")
        emit_array(fh, "nm_font_packed", list(packed))
        fh.write("\n")
        emit_array(fh, "nm_font_fallback", list(FALLBACK))
        fh.write("""
/* 输入：位偏移；输出：该位置起 NM_FONT_INK 位的值（MSB 在前）。
 * 预期行为：字形按 49 位一组紧密排列、不是字节对齐，所以只能按位取；固定读「当前字节 +
 *           下一个字节」，表尾的 0 补位保证最后一个字也不会越界。 */
static uint32_t nm_font_read7(uint32_t bitpos)
{
    uint32_t b = bitpos >> 3;
    uint32_t w = ((uint32_t)nm_font_packed[b] << 8) | (uint32_t)nm_font_packed[b + 1];

    return (w >> (9u - (bitpos & 7u))) & 0x7Fu;
}

void nm_font_glyph(uint32_t cp, uint8_t *out)
{
    const uint8_t *p = nm_font_cp;
    uint32_t cur = 0;
    int i, r;

    for (i = 0; i < NM_FONT_N; i++) {
        uint32_t delta = 0;
        int shift = 0;
        uint8_t byte;

        do {
            byte = *p++;
            delta |= (uint32_t)(byte & 0x7F) << shift;
            shift += 7;
        } while (byte & 0x80);

        cur += delta;
        if (cur > cp)
            break;                      /* 码点升序，后面的只会更大 */
        if (cur == cp) {
            uint32_t bit = (uint32_t)i * NM_FONT_GLYPH_BITS;
            for (r = 0; r < NM_FONT_ROW; r++, bit += NM_FONT_INK)
                out[r] = (uint8_t)(nm_font_read7(bit) << 1);
            return;
        }
    }
    for (r = 0; r < NM_FONT_ROW; r++)
        out[r] = nm_font_fallback[r];
}

void nm_font_glyph_utf8(const uint8_t *s, int n, uint8_t *out)
{
    int need = nm_utf8_seq_len(s[0]);
    int r;

    if (need == 0 || n < need || !nm_utf8_valid(s, need)) {
        for (r = 0; r < NM_FONT_ROW; r++)
            out[r] = nm_font_fallback[r];
        return;
    }
    nm_font_glyph(nm_utf8_decode(s, need), out);
}
""")


def main():
    """输入：无；输出：infer/c/nm_font.h 与 nm_font.c。预期行为：写完回读比对。"""
    glyphs = read_subset(SUBSET)
    n = len(glyphs)
    cp_bytes = build_cp_table(glyphs)
    packed = pack_glyphs(glyphs)

    if parse_cp_back(cp_bytes, n) != [cp for cp, _ in glyphs]:
        raise SystemExit("自检失败：生成的码点表解回来与子集清单不一致")
    if unpack_glyphs(packed, n) != [blob for _, blob in glyphs]:
        raise SystemExit("自检失败：生成的点阵位流解回来与子集清单不一致")

    write_header(n, cp_bytes, packed)
    write_source(n, cp_bytes, packed)

    print("写入 %s：%d 字，点阵 %d B（打包，原 %d B） + 码点表 %d B = %d B"
          % (OUT_C.name, n, len(packed), n * ROW_BYTES, len(cp_bytes),
             len(packed) + len(cp_bytes)))


if __name__ == "__main__":
    main()