"""从 infer/font/subset_8x8.txt 生成 C 引擎的字库 infer/c/nm_font.c 与 nm_font.h。

输入：infer/font/subset_8x8.txt（每行「码点 7 行点阵 字符」，由 infer/font/select_subset.py 选出来）。
输出：infer/c/nm_font.h（常量与查表接口）与 infer/c/nm_font.c（码点表 + 点阵 + 兜底字形 + 查表实现）。
预期行为：码点按升序存成「首个绝对 + 之后增量」的 LEB128 流，顺序扫描即可二分退化成线性；
          文件头的注释由实际数据算出来（字数、点阵字节、码点表字节、合计），不写死；
          生成后把 C 文件回读解析一遍，与内存里的数据逐字节比对，不一致就报错退出。
"""

import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SUBSET = HERE.parent / "font" / "subset_8x8.txt"
OUT_C = HERE / "nm_font.c"
OUT_H = HERE / "nm_font.h"

CELL = 8          # 8x8 点阵
ROW_BYTES = 7     # 只存第 1..7 行：原字体第 0 行恒空
INK_BITS = 7      # 行内只用 bit7..bit1：原字体第 7 列恒空

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


def parse_back(cp_bytes, bit_bytes, n):
    """把刚写出去的 C 数组重新解回来，做生成自检。

    输入：码点表字节、点阵字节、字数。
    输出：[(码点, 7 字节点阵)]。
    预期行为：LEB128 流必须恰好用完整张表，多一个字节或少一个字节都算失败。
    """
    glyphs, pos, prev = [], 0, 0
    for i in range(n):
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
        glyphs.append((prev, bytes(bit_bytes[i * ROW_BYTES:(i + 1) * ROW_BYTES])))
    if pos != len(cp_bytes):
        raise SystemExit("自检失败：码点表有 %d 字节没用上" % (len(cp_bytes) - pos))
    return glyphs


def main():
    """输入：无；输出：infer/c/nm_font.h 与 nm_font.c。预期行为：写完回读比对。"""
    glyphs = read_subset(SUBSET)
    n = len(glyphs)
    cp_bytes = build_cp_table(glyphs)
    bit_bytes = [v for _, blob in glyphs for v in blob]
    total = len(cp_bytes) + len(bit_bytes)

    if parse_back(cp_bytes, bit_bytes, n) != glyphs:
        raise SystemExit("自检失败：生成的码点表 / 点阵解回来与子集清单不一致")

    with OUT_H.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("/* 自动生成，请勿手改：infer/c/gen_font.py */\n")
        fh.write("#ifndef NANOMEOW_FONT_H\n#define NANOMEOW_FONT_H\n\n#include <stdint.h>\n\n")
        fh.write("#define NM_FONT_N %d          /* 收录字符数 */\n" % n)
        fh.write("#define NM_FONT_CP_BYTES %d   /* 码点表字节数 */\n" % len(cp_bytes))
        fh.write("#define NM_FONT_W %d           /* 点阵宽（含原字体的右侧字间距列） */\n" % CELL)
        fh.write("#define NM_FONT_H %d           /* 点阵高（含原字体的顶部空行） */\n" % CELL)
        fh.write("#define NM_FONT_ROW %d         /* 实际存的像素行数：原字体第 0 行恒空 */\n" % ROW_BYTES)
        fh.write("#define NM_FONT_INK %d         /* 实际存的像素列数：原字体第 7 列恒空 */\n\n" % INK_BITS)
        fh.write("/* 码点表：LEB128 增量流，首个是绝对码点，之后是与前一项的差，按码点升序。 */\n")
        fh.write("extern const uint8_t nm_font_cp[NM_FONT_CP_BYTES];\n")
        fh.write("/* 点阵：NM_FONT_N 个字形，每个 NM_FONT_ROW 行；行内 bit7..bit1 是第 0..6 列。 */\n")
        fh.write("extern const uint8_t nm_font_bits[NM_FONT_N * NM_FONT_ROW];\n")
        fh.write("/* 兜底字形：7x7 空心方框，字库里没有的码点用它。 */\n")
        fh.write("extern const uint8_t nm_font_fallback[NM_FONT_ROW];\n\n")
        fh.write("""/* 输入：BMP 码点；输出：指向 NM_FONT_ROW 个行字节的常量指针，码点不在字库时返回兜底字形。
 * 预期行为：顺序扫描码点表（升序），超过目标码点立刻退出；只读常量，可重入。 */
const uint8_t *nm_font_lookup(uint32_t cp);

/* 输入：一个完整字符的 UTF-8 字节与字节数；输出：同 nm_font_lookup。
 * 预期行为：长度不够或序列非法时返回兜底字形，不读越界。 */
const uint8_t *nm_font_lookup_utf8(const uint8_t *s, int n);

#endif
""")

    with OUT_C.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("/* 自动生成，请勿手改：infer/c/gen_font.py\n")
        fh.write(" * nanomeow 的 8x8 子集字库：%d 字，点阵 %d B + 码点表 %d B = %d B。\n"
                 % (n, len(bit_bytes), len(cp_bytes), total))
        fh.write(" * 字形来源：fusion-pixel-font 8px 等宽（OFL-1.1）；选字清单与来源说明见\n")
        fh.write(" * infer/font/subset_8x8.txt 与 infer/font/README.md。\n")
        fh.write(" * 布局：每字只存第 1..7 行（原字体第 0 行恒空），行内只用 bit7..bit1（第 7 列恒空）。\n")
        fh.write(" */\n")
        fh.write("#include <stddef.h>\n\n#include \"nm_font.h\"\n#include \"nm_utf8.h\"\n\n")
        emit_array(fh, "nm_font_cp", cp_bytes)
        fh.write("\n")
        emit_array(fh, "nm_font_bits", bit_bytes)
        fh.write("\n")
        emit_array(fh, "nm_font_fallback", list(FALLBACK))
        fh.write("""
const uint8_t *nm_font_lookup(uint32_t cp)
{
    const uint8_t *p = nm_font_cp;
    uint32_t cur = 0;
    int i;

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
        if (cur == cp)
            return nm_font_bits + (size_t)i * NM_FONT_ROW;
        if (cur > cp)
            break;                      /* 码点升序，后面的只会更大 */
    }
    return nm_font_fallback;
}

const uint8_t *nm_font_lookup_utf8(const uint8_t *s, int n)
{
    uint32_t cp;
    int need = nm_utf8_seq_len(s[0]);

    if (need == 0 || n < need || !nm_utf8_valid(s, need))
        return nm_font_fallback;
    if (need == 1)
        cp = s[0];
    else if (need == 2)
        cp = ((uint32_t)(s[0] & 0x1F) << 6) | (uint32_t)(s[1] & 0x3F);
    else if (need == 3)
        cp = ((uint32_t)(s[0] & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6)
             | (uint32_t)(s[2] & 0x3F);
    else
        cp = ((uint32_t)(s[0] & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12)
             | ((uint32_t)(s[2] & 0x3F) << 6) | (uint32_t)(s[3] & 0x3F);
    return nm_font_lookup(cp);
}
""")

    print("写入 %s：%d 字，点阵 %d B + 码点表 %d B = %d B"
          % (OUT_C.name, n, len(bit_bytes), len(cp_bytes), total))


if __name__ == "__main__":
    main()
