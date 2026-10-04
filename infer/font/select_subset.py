"""从 fusion-pixel-font 的 8px 等宽 BDF 里选出 nanomeow 要用的 8x8 子集字库。

输入：--bdf-dir 指向解压后的 8px 等宽 BDF 目录（含 latin 与 zh_hans 两份），
      --corpus 指向训练语料 jsonl（每行 {"text": "user:...\nbot:..."}），
      --budget 是点阵数据 + 码点表的字节上限。
输出：--out（默认 infer/font/subset_8x8.txt），每行「码点 7 行点阵 字符」。
预期行为：只取语料 bot 侧的字符，按词频降序装到装不下为止；字形按基线摆进 8x8 格后
          只留 7x7 —— 实测这批字形第 0 行与第 7 列恒空，所以不存；
          摆格后如果哪一行越界或第 7 列非空，直接报错退出（口径变了要人来拍板）。
"""

import argparse
import json
import pathlib
from collections import Counter

CELL = 8                      # 8x8 点阵
ROW_BYTES = CELL - 1          # 只存第 1..7 行，每行 1 字节（bit7..bit1 是第 0..6 列）


def parse_bdf(path):
    """解析 BDF。

    输入：BDF 文件路径。
    输出：(ascent, {码点: (BBX, 行十六进制字符串列表)})。
    预期行为：只认 ENCODING >= 0 的字形，未映射字形直接跳过。
    """
    ascent = 0
    glyphs = {}
    enc, bbx, rows, in_bitmap = None, None, [], False
    for line in path.open("r", encoding="latin-1"):
        line = line.rstrip("\n")
        if line.startswith("FONT_ASCENT"):
            ascent = int(line.split()[1])
        elif line.startswith("STARTCHAR"):
            enc, bbx, rows = None, None, []
        elif line.startswith("ENCODING"):
            enc = int(line.split()[1])
        elif line.startswith("BBX"):
            bbx = tuple(int(v) for v in line.split()[1:5])
        elif line.startswith("BITMAP"):
            in_bitmap = True
        elif line.startswith("ENDCHAR"):
            in_bitmap = False
            if enc is not None and enc >= 0 and bbx is not None:
                glyphs[enc] = (bbx, list(rows))
        elif in_bitmap:
            rows.append(line)
    if ascent == 0:
        raise SystemExit("自检失败：%s 里没有 FONT_ASCENT" % path.name)
    return ascent, glyphs


def to_cell(bbx, rows, ascent):
    """把 BDF 字形摆进 8x8 格。

    输入：BBX 四元组、行十六进制字符串、字体 ascent。
    输出：8 个 int 的列表，第 0 项是最上面一行，每项 bit7 是最左列。
    预期行为：按基线对齐；超出格子的像素直接报错（说明这批字形不能用 7x7 存）。
    """
    w, h, xoff, yoff = bbx
    cell = [0] * CELL
    top = ascent - (yoff + h)
    for i, hex_row in enumerate(rows):
        r = top + i
        if not (0 <= r < CELL):
            raise SystemExit("自检失败：字形有像素落在 8x8 格之外（行 %d）" % r)
        val = int(hex_row, 16) if hex_row else 0
        val >>= max(0, w - CELL)
        if xoff > 0:
            val <<= xoff
        elif xoff < 0:
            val >>= -xoff
        cell[r] = val & 0xFF
    return cell


def packed(cell):
    """把 8x8 格压成 7 字节。

    输入：to_cell 的输出。
    输出：7 字节 bytes，丢掉恒空的第 0 行与恒空的第 7 列（bit0）。
    预期行为：第 0 行或第 7 列非空时报错退出，不静默丢像素。
    """
    if cell[0] != 0:
        raise SystemExit("自检失败：第 0 行非空，7x7 存法不成立")
    if any(v & 1 for v in cell):
        raise SystemExit("自检失败：第 7 列非空，7x7 存法不成立")
    return bytes(cell[1:])


def varint_size(values):
    """输入：已排序的码点列表；输出：增量 LEB128 编码后的总字节数。

    预期行为：首个是绝对码点，之后每项是与前一项的差；不产出数据，只算长度。
    """
    total, prev = 0, 0
    for v in values:
        d = v - prev
        prev = v
        total += 1
        while d >= 0x80:
            d >>= 7
            total += 1
    return total


def bot_chars(corpus):
    """输入：语料路径；输出：bot 侧字符的 Counter（按出现次数）。

    预期行为：模板是 `user:<内容>\nbot:<内容>`，冒号后没有空格；缺 `\nbot:` 的行报错。
    """
    cnt = Counter()
    with corpus.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            text = json.loads(line)["text"]
            if "\nbot:" not in text:
                raise SystemExit("自检失败：%s:%d 没有 `\\nbot:` 分隔" % (corpus.name, lineno))
            cnt.update(text.split("\nbot:", 1)[1])
    return cnt


def select(freq, glyphs, ascent, budget):
    """按词频降序选字，装到预算装不下为止。

    输入：字符 Counter、BDF 字形表、ascent、字节上限。
    输出：(选中的 [(码点, 7 字节)] 按码点升序, 因为缺字形或超预算而放弃的字符列表)。
    预期行为：每加一个字都重算码点表长度，保证「点阵 + 码点表」一起不超预算。
    """
    chosen, skipped, missing = [], [], []
    for ch, _ in freq.most_common():
        cp = ord(ch)
        if cp not in glyphs:
            missing.append(ch)
            continue
        blob = packed(to_cell(*glyphs[cp], ascent))
        cps = sorted([c for c, _ in chosen] + [cp])
        if len(cps) * ROW_BYTES + varint_size(cps) > budget:
            skipped.append(ch)
            continue
        chosen.append((cp, blob))
    chosen.sort(key=lambda item: item[0])
    return chosen, skipped, missing


def main():
    """输入：命令行参数；输出：subset_8x8.txt。预期行为：写前逐字回读自检。"""
    ap = argparse.ArgumentParser(description="选出 nanomeow 的 8x8 子集字库")
    ap.add_argument("--bdf-dir", type=pathlib.Path,
                    default=pathlib.Path("tmp/f8bdf"),
                    help="8px 等宽 BDF 目录（含 latin 与 zh_hans 两份）")
    ap.add_argument("--corpus", type=pathlib.Path,
                    default=pathlib.Path("train/data/chitchat_para.jsonl"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("infer/font/subset_8x8.txt"))
    ap.add_argument("--budget", type=int, default=2900,
                    help="点阵 + 码点表的字节上限；默认值按「整机 Flash 余 3,480 B，"
                         "扣掉查表代码 344 B、.ARM.exidx 16 B，再给 OLED 驱动留 ~220 B」定的")
    args = ap.parse_args()

    ascent, glyphs = None, {}
    for name in ("fusion-pixel-8px-monospaced-latin.bdf",
                 "fusion-pixel-8px-monospaced-zh_hans.bdf"):
        path = args.bdf_dir / name
        if not path.exists():
            raise SystemExit("缺少 %s（见 infer/font/README.md 的下载步骤）" % path)
        asc, g = parse_bdf(path)
        ascent = asc if ascent is None else ascent
        if asc != ascent:
            raise SystemExit("自检失败：两份 BDF 的 FONT_ASCENT 不一致")
        glyphs.update(g)

    freq = bot_chars(args.corpus)
    chosen, skipped, missing = select(freq, glyphs, ascent, args.budget)
    if not chosen:
        raise SystemExit("一个字都没选中，预算太小？")
    cps = [cp for cp, _ in chosen]
    total = len(chosen) * ROW_BYTES + varint_size(cps)

    lines = [
        "# nanomeow 8x8 子集字库（自动生成，请勿手改）：infer/font/select_subset.py",
        "# 来源：fusion-pixel-font 8px 等宽 zh_hans + latin（OFL-1.1），见同目录 LICENSE-OFL.txt",
        "# 选字：%s 的 bot 侧字符，按词频降序，字节上限 %d" % (args.corpus.as_posix(), args.budget),
        "# 点阵：每字 7 行（原字体第 0 行与第 7 列恒空，不存），行内 bit7..bit1 是第 0..6 列",
        "# 每行：<码点十六进制> <7 行点阵十六进制> <字符>",
        "# 收录 %d 字 / %d 字节；因超预算放弃 %d 字；因 BDF 缺字形放弃 %d 字"
        % (len(chosen), total, len(skipped), len(missing)),
    ]
    for cp, blob in chosen:
        lines.append("%04X %s %s" % (cp, blob.hex().upper(), chr(cp)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    # 回读自检：写出去的东西必须能原样解回来
    back = {}
    for line in args.out.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        cp_hex, blob_hex, _ = line.split(" ")
        back[int(cp_hex, 16)] = bytes.fromhex(blob_hex)
    if back != dict(chosen):
        raise SystemExit("自检失败：%s 回读结果与内存里不一致" % args.out)
    print("写入 %s：%d 字 / %d 字节（预算 %d，放弃 %d 字）"
          % (args.out.as_posix(), len(chosen), total, args.budget, len(skipped) + len(missing)))


if __name__ == "__main__":
    main()
