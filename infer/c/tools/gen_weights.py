"""从 infer/model_weights.h 生成 C 引擎的权重描述表 infer/c/generated/nm_weights.c。

输入：infer/model_weights.h（由 train/scripts/export_int8.py 生成）、infer/c/engine/nanomeow.h 与
      infer/c/engine/nanomeow.c（用来统计引擎的运行期内存）。
输出：infer/c/generated/nm_weights.c（`nm_blocks[3]` 与四个顶层 `nm_mat` 的定义）。
预期行为：把「int8 码 + 每行乘子 + 每行移位」**无损压缩**后装成 `nm_mat` ——
          头文件里 `<符号>_mul` 是数组的是 per-row（rows = 乘子条数、cols = 码数 / rows），
          是标量的是 per-tensor（rows = 1、cols = 码数）。
          压缩有两条，都能反解回原值（见 nanomeow.c 的 nm_row_scale / nm_row_codes）：
          ① scale：尾数只存 23 位（最高位恒 1），移位按张量存 1 字节基线 + 行内小位宽增量；
          ② 码：整张量行去重，只存唯一行池 + 每行索引（索引表跟在 scale 位流后面）。
          文件头的注释由实际参数统计出来：每个导出权重的名字与字节数、权重合计、
          Flash 常驻表大小、RAM（状态 + 引擎工作区）大小，全部现算，不写死。
          生成前自检：3 层每个字段都找得到、码数能被行数整除，否则报错退出。
"""

import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
HEADER = HERE.parents[1] / "model_weights.h"
ENGINE_H = HERE.parent / "engine" / "nanomeow.h"
ENGINE_C = HERE.parent / "engine" / "nanomeow.c"
OUT = HERE.parent / "generated" / "nm_weights.c"
CFG = HERE.parents[1] / "model_cfg.h"    # 结构常量的唯一来源，由导出脚本按 checkpoint 生成
N_LAYER = 3

# nm_block 字段名 -> 头文件符号里 `nm_blocks_<层号>_` 之后的那一段
BLOCK_FIELDS = (
    ("ln0_weight", "ln0_weight"),
    ("ln0_bias", "ln0_bias"),
    ("ln1_weight", "ln1_weight"),
    ("ln1_bias", "ln1_bias"),
    ("ln2_weight", "ln2_weight"),
    ("ln2_bias", "ln2_bias"),
    ("x_r", "att_x_r"),
    ("x_w", "att_x_w"),
    ("x_k", "att_x_k"),
    ("x_v", "att_x_v"),
    ("x_a", "att_x_a"),
    ("x_g", "att_x_g"),
    ("w1", "att_w1"),
    ("w2", "att_w2"),
    ("w0", "att_w0"),
    ("a1", "att_a1"),
    ("a2", "att_a2"),
    ("a0", "att_a0"),
    ("v1", "att_v1"),
    ("v2", "att_v2"),
    ("v0", "att_v0"),
    ("g1", "att_g1"),
    ("g2", "att_g2"),
    ("k_k", "att_k_k"),
    ("k_a", "att_k_a"),
    ("r_k", "att_r_k"),
    ("receptance", "att_receptance_weight"),
    ("key", "att_key_weight"),
    ("value", "att_value_weight"),
    ("output", "att_output_weight"),
    ("ln_x_weight", "att_ln_x_weight"),
    ("ln_x_bias", "att_ln_x_bias"),
    ("ffn_x_k", "ffn_x_k"),
    ("ffn_key", "ffn_key_weight"),
    ("ffn_value", "ffn_value_weight"),
)

# 只有第 0 层有 ln0（训练侧 `blocks.0.ln0`），其余层没有对应符号，填空矩阵
LAYER0_ONLY = ("ln0_weight", "ln0_bias")

# 引擎侧的 nm_mat 名字 -> 头文件里的原始符号（后者带 nmw_ 前缀，两者不能同名）
TOP_FIELDS = (("nm_emb_weight", "nmw_emb_weight"),
              ("nm_ln_out_weight", "nmw_ln_out_weight"),
              ("nm_ln_out_bias", "nmw_ln_out_bias"),
              ("nm_head_weight", "nmw_head_weight"))

EMPTY_MAT = "{0, 0, 0, 0, 0, 0}"


def int_list(text, sym):
    """输入：头文件文本、符号名；输出：该符号的整数列表（数组与标量都归一成 list），
    符号不存在时返回 None。预期行为：与 parse_header 用同一套正则，只是多取了值。"""
    m = re.search(r"static const \w+ %s\[\] = \{([^{}]*)\};" % re.escape(sym), text)
    if m:
        return [int(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
    m = re.search(r"static const \w+ %s = (-?\d+);" % re.escape(sym), text)
    if m:
        return [int(m.group(1))]
    return None


MANT_BITS = 23      # 归一化后尾数最高位恒 1，只存低 23 位
SHIFT_BITS = 7      # shift ∈ [-127, 0]，存 shift + 127


def mantissa(mul, sym):
    """输入：乘子、符号名；输出：归一化尾数（23 位，去掉恒 1 的最高位）。

    预期行为：mul 实测恒在 [2^30, 2^31) 且至少有 7 个尾零，所以 mul = (2^23 + mant) << 7。
              前提不成立直接报错，不静默截断。
    """
    if mul <= 0 or mul % 128:
        raise SystemExit("自检失败：%s 的 mul=%d 不是 128 的正整数倍" % (sym, mul))
    m24 = mul >> 7
    if not (1 << 23) <= m24 < (1 << 24):
        raise SystemExit("自检失败：%s 的 mul=%d 归一化后不在 [2^23, 2^24)" % (sym, mul))
    return m24 - (1 << 23)


def shift_value(sh, sym):
    """输入：移位、符号名；输出：原值。预期行为：只接受 [-127, 0]，否则报错。"""
    if not -127 <= sh <= 0:
        raise SystemExit("自检失败：%s 的 shift=%d 超出 [-127, 0]" % (sym, sh))
    return sh


def bit_pack(values, width):
    """输入：非负整数列表、位宽；输出：小端位序的字节串（最后 1 字节高位补 0）。

    预期行为：第 i 个值占位 [i*width, (i+1)*width)，与 nanomeow.c 的 nm_read_bits 互逆。
              width = 0 时输出空串。
    """
    if width == 0:
        return b""
    out, acc, nbits = bytearray(), 0, 0
    for v in values:
        acc |= (v & ((1 << width) - 1)) << nbits
        nbits += width
        while nbits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8
    if nbits:
        out.append(acc & 0xFF)
    return bytes(out)


def pack_scale_tensor(mul, sh, sym):
    """输入：乘子、移位、符号名；输出：per-tensor 的 4 B scale 位流。

    预期行为：小端 32 位 = (尾数 23 位) << 7 | (shift + 127)，与 nm_row_scale 的
              per-tensor 分支互逆。
    """
    v = (mantissa(mul, sym) << SHIFT_BITS) | (shift_value(sh, sym) + 127)
    return bytes([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF])


def pack_scale_rows(muls, shifts, sym):
    """输入：逐行乘子、逐行移位、符号名；输出：per-row 的 scale 位流。

    预期行为：布局 = [尾数 23 位打包][1 B 基线 shift][1 B 增量位宽 w][增量 w 位打包]。
              基线取整张量最小的 shift，增量非负；整张量同一个 shift 时 w = 0、不存增量。
              与 nm_row_scale 的 per-row 分支互逆。
    """
    if len(muls) != len(shifts):
        raise SystemExit("自检失败：%s 的 mul(%d) 与 shift(%d) 个数不一致"
                         % (sym, len(muls), len(shifts)))
    mants = [mantissa(mu, sym) for mu in muls]
    shs = [shift_value(sh, sym) for sh in shifts]
    base = min(shs)
    deltas = [sh - base for sh in shs]
    w = max(deltas).bit_length()
    if w > 8:
        raise SystemExit("自检失败：%s 的移位增量位宽 %d 超过 8" % (sym, w))
    return (bit_pack(mants, MANT_BITS) + bytes([base & 0xFF, w])
            + bit_pack(deltas, w))


def plan_rows(codes, rows, cols):
    """输入：int8 码、行数、列数；输出：(唯一行池, 索引位宽, 索引列表)，不去重时 None。

    预期行为：唯一行数 uniq < rows 且索引表字节数严格小于省下的码字节数才去重；
              索引位宽 b = (uniq - 1).bit_length()，索引表跟在 scale 位流后面。
    """
    if rows < 2 or cols < 2:
        return None
    pool, seen, idx = [], {}, []
    for r in range(rows):
        row = tuple(codes[r * cols:(r + 1) * cols])
        if row not in seen:
            seen[row] = len(pool)
            pool.append(row)
        idx.append(seen[row])
    uniq = len(pool)
    if uniq == rows:
        return None
    b = max(1, (uniq - 1).bit_length())   # 0 位会被运行期当成「没去重」，所以至少 1 位
    if (rows * b + 7) // 8 >= (rows - uniq) * cols:
        return None
    flat = [v for row in pool for v in row]
    return flat, b, idx


def parse_header(text):
    """输入：model_weights.h 文本；输出：(每个符号的元素个数, 每个符号是不是数组)。

    预期行为：数组认 `static const <类型> <名字>[] = {...};`（元素个数 = 顶层逗号数 + 1，
              初始化列表里没有嵌套花括号），标量认 `static const <类型> <名字> = <整数>;`。
              返回的数组顺序 = 文件里的顺序。
    """
    lens, is_array, order = {}, {}, []
    for m in re.finditer(r"static const \w+ (nmw_\w+?)(\[\])? = \{([^{}]*)\};", text):
        name, brackets, body = m.group(1), m.group(2), m.group(3)
        lens[name] = body.count(",") + 1
        is_array[name] = brackets is not None
        order.append(name)
    for m in re.finditer(r"static const \w+ (nmw_\w+?) = -?\d+;", text):
        lens[m.group(1)] = 1
        is_array[m.group(1)] = False
    return lens, is_array, order


def is_tensor(sym, lens):
    """输入：符号名、元素个数表；输出：它是不是一个张量（而不是张量的 _mul / _shift 分量）。"""
    for suffix in ("_mul", "_shift"):
        if sym.endswith(suffix) and sym[: -len(suffix)] in lens:
            return False
    return True


def display_name(sym):
    """输入：符号名；输出：训练侧参数名。

    预期行为：`nmw_` 前缀去掉后把 `blocks_<i>_att_` / `blocks_<i>_ffn_` 还原成
              `blocks.<i>.att.` / `blocks.<i>.ffn.`，末尾的 `_weight` / `_bias` 还原成
              `.weight` / `.bias`（与 export_int8.py 的 `nmw_ + name.replace(".", "_")` 互逆）。
    """
    rest = sym[len("nmw_"):]
    prefix = ""
    m = re.match(r"blocks_(\d+)_(.+)$", rest)
    if m:
        prefix, rest = "blocks.%s." % m.group(1), m.group(2)
        for head in ("att_", "ffn_"):
            if rest.startswith(head):
                prefix, rest = prefix + head[:-1] + ".", rest[len(head):]
                break
    for tail in ("_weight", "_bias"):
        if rest.endswith(tail):
            rest = rest[: -len(tail)] + "." + tail[1:]
            break
    return prefix + rest


def mat_shape(sym, lens, is_array, idx_bits):
    """输入：符号名、元素个数表、数组标记表、索引位宽；输出：(rows, cols, per_row, idx_bits)。

    预期行为：per-row 张量用头文件声明的行数，per-tensor 张量视作 1 行；码数不是行数整数倍时报错。
    """
    if sym not in lens:
        raise SystemExit("自检失败：头文件里没有符号 %s" % sym)
    n = lens[sym]
    if is_array.get(sym + "_mul", False):
        rows = lens[sym + "_mul"]
        if rows <= 0 or n % rows:
            raise SystemExit("自检失败：%s 的码数 %d 不是行数 %d 的整数倍" % (sym, n, rows))
        return rows, n // rows, 1, idx_bits
    return 1, n, 0, 0


def mat_expr(shape, code_off, scale_off):
    """输入：(rows, cols, per_row, idx_bits) 与权重池里的码 / scale 偏移；输出：一个 nm_mat 表达式。"""
    rows, cols, per_row, idx_bits = shape
    return "{%d, %d, %d, %d, %d, %d}" % (code_off, scale_off, rows, cols, per_row, idx_bits)


def parse_defines(text):
    """输入：infer/model_cfg.h 文本；输出：{NM_X: 整数}。

    预期行为：只认 `#define NMW_X <十进制>`，返回时把 NMW_ 前缀换成 NM_。结构常量由
              train/scripts/export_int8.py 按 checkpoint 生成，这里跟着它走，不再抄一份。
    """
    return {"NM_" + m.group(1): int(m.group(2))
            for m in re.finditer(r"^#define NMW_(\w+) (\d+)$", text, re.M)}


def state_bytes(d):
    """输入：宏表；输出：单层 nm_layer_state 的字节数。

    预期行为：镜像 infer/c/engine/nanomeow.h 里 nm_layer_state 的布局 ——
              int8[NM_N_EMBD] + nm_scale + int 各两份（time-shift 的上一 token），
              再接 int32[n_head][head_size][head_size] + nm_scale + int。
              nm_scale 是 { int32_t m; int8_t e; }，按 4 字节对齐后占 8 字节。
    """
    scale = 8
    prev = d["NM_N_EMBD"] + scale + 4
    wkv = d["NM_N_HEAD"] * d["NM_HEAD_SIZE"] * d["NM_HEAD_SIZE"] * 4
    return 2 * prev + wkv + scale + 4


def workspace_bytes(engine, d):
    """输入：nanomeow.c 文本、宏表；输出：(工作区字节数, 暂存张量个数)。

    预期行为：统计 `static nm_tensor a, b, c;` 的个数（每个 = int64[NM_MAX_DIM] + nm_scale，
              而 NM_MAX_DIM = NM_DIM_FFN），再加上 `static int32_t s_w15[NM_N_EMBD];` 的
              Q15 衰减表。工作区是常驻 RAM 的引擎暂存，不含栈。
    """
    if "#define NM_MAX_DIM NM_DIM_FFN" not in engine:
        raise SystemExit("自检失败：nanomeow.c 里 NM_MAX_DIM 不再是 NM_DIM_FFN，工作区口径要重算")
    n = 0
    for m in re.finditer(r"^static nm_tensor ([^;]+);", engine, re.M):
        n += len([x for x in m.group(1).split(",") if x.strip()])
    tensor = d["NM_DIM_FFN"] * 8 + 8
    total = n * tensor + d["NM_N_EMBD"] * 4
    for m in re.finditer(r"^static int64_t \w+\[NM_VOCAB\];", engine, re.M):
        total += d["NM_VOCAB"] * 8
    return total, n


# 引擎读不到的张量：nanomeow.c 在 `if (layer == 0)` 分支里直接取 v_first，value 残差的
# v0 / v1 / v2 三个张量对第 0 层根本不参与计算（infer/ref/model.py 的 `if i == 0` 同样跳过，
# 参考实现 Mini_RWKV_7 的第 0 层也没有 value 残差）。它们的码与 scale 因此不进权重池，
# 描述符留空并在生成文件里写明原因 —— 去掉的是「引擎读不到的字节」，不是改架构：
# 第 1 / 2 层的同名张量照旧完整导出，G1 逐位对拍仍是同一份数值。
ENGINE_UNREACHABLE = frozenset((
    "nmw_blocks_0_att_v0", "nmw_blocks_0_att_v1", "nmw_blocks_0_att_v2",
))


def main():
    """输入：无；输出：写 nm_weights.c 并在 stdout 打印一行摘要。"""
    hdr = HEADER.read_text(encoding="utf-8")
    lens, is_array, order = parse_header(hdr)
    d = parse_defines(CFG.read_text(encoding="utf-8"))
    work_bytes, work_n = workspace_bytes(ENGINE_C.read_text(encoding="utf-8"), d)

    tensors = [s for s in order if is_tensor(s, lens)]
    blobs, pools, idxbits = {}, {}, {}
    code_total = 0
    for sym in tensors:
        if sym in ENGINE_UNREACHABLE:
            continue
        muls = int_list(hdr, sym + "_mul")
        shifts = int_list(hdr, sym + "_shift")
        codes = int_list(hdr, sym)
        if muls is None or shifts is None:
            raise SystemExit("自检失败：%s 没有 mul/shift 分量" % sym)
        if codes is None or len(codes) != lens[sym]:
            raise SystemExit("自检失败：%s 的码数与头文件声明不一致" % sym)
        rows, n = len(muls), lens[sym]
        if rows <= 0 or n % rows:
            raise SystemExit("自检失败：%s 的码数 %d 不是行数 %d 的整数倍" % (sym, n, rows))
        cols = n // rows
        idxbits[sym] = 0
        if is_array.get(sym + "_mul", False):
            blob = pack_scale_rows(muls, shifts, sym)
            plan = plan_rows(codes, rows, cols)
            if plan is None:
                code_total += n
            else:
                flat, b, idx = plan
                blob += bit_pack(idx, b)
                pools[sym] = flat
                idxbits[sym] = b
                code_total += len(flat)
        else:
            blob = pack_scale_tensor(muls[0], shifts[0], sym)
            code_total += n
        blobs[sym] = blob

    # 所有码与 scale 位流拼进一个池：描述符只存 uint16 偏移，12 B -> 8 B（109 个共省 436 B）。
    pool, offs = bytearray(), {}
    for sym in tensors:
        if sym in ENGINE_UNREACHABLE:
            offs[sym] = (0, 0, 0, 0, 0, 0)
            continue
        codes = pools[sym] if sym in pools else int_list(hdr, sym)
        code_off = len(pool)
        pool += bytes(v & 0xFF for v in codes)
        scale_off = len(pool)
        pool += blobs[sym]
        offs[sym] = (code_off, scale_off) + mat_shape(sym, lens, is_array, idxbits[sym])
    if len(pool) > 65535:
        raise SystemExit("自检失败：权重池 %d B 超过 uint16 偏移上限 65535" % len(pool))
    scale_total = sum(len(blobs[s]) for s in tensors if s in blobs)
    state_total = state_bytes(d) * d["NM_N_LAYER"]

    head = ["/* 自动生成，请勿手改：infer/c/tools/gen_weights.py",
            " *",
            " * 权重来源：infer/model_weights.h（train/scripts/export_int8.py 从 checkpoint 导出）",
            " * 模型：%d 层 RWKV-7 x070 缩维，n_embd=%d，n_head=%d，head_size=%d，"
            "dim_ffn=%d，词表 %d"
            % (d["NM_N_LAYER"], d["NM_N_EMBD"], d["NM_N_HEAD"], d["NM_HEAD_SIZE"],
               d["NM_DIM_FFN"], d["NM_VOCAB"]),
            " *",
            " * 导出的权重（名字 = 训练侧参数名；字节数 = int8 码 / scale 表）："]
    for sym in tensors:
        if sym in ENGINE_UNREACHABLE:
            head.append(" *   %-32s      -           -  (第 0 层不做 value 残差，引擎读不到，不导出)"
                        % display_name(sym))
            continue
        if is_array.get(sym + "_mul", False):
            rows = lens[sym + "_mul"]
            kind = "per-row %d x %d" % (rows, lens[sym] // rows)
            stored = lens[sym]
            if sym in pools:
                stored = len(pools[sym])
                kind += "，去重成 %d 行" % (stored // (lens[sym] // rows))
        else:
            kind = "per-tensor %d" % lens[sym]
            stored = lens[sym]
        head.append(" *   %-32s %6d B / %5d B  (%s)"
                    % (display_name(sym), stored, len(blobs[sym]), kind))
    head += [
        " *",
        " * 权重合计：码 %d B（%.1f KiB）+ scale 表 %d B（%.1f KiB）= %d B（%.1f KiB）"
        % (code_total, code_total / 1024.0, scale_total, scale_total / 1024.0,
           code_total + scale_total, (code_total + scale_total) / 1024.0),
        " * 内存占用（Flash 常驻只读表）：%d B（%.1f KiB）"
        % (code_total + scale_total, (code_total + scale_total) / 1024.0),
        " * 内存占用（RAM）：状态 %d 层 x %d B = %d B（%.1f KiB，含每层 int32 wkv %d B）"
        % (d["NM_N_LAYER"], state_bytes(d), state_total, state_total / 1024.0,
           d["NM_N_HEAD"] * d["NM_HEAD_SIZE"] * d["NM_HEAD_SIZE"] * 4),
        " *                  + 引擎工作区 %d 个 int64[%d] 暂存 + Q15 衰减表 = %d B（%.1f KiB）"
        % (work_n, d["NM_DIM_FFN"], work_bytes, work_bytes / 1024.0),
        " *                  = %d B（%.1f KiB），不含调用栈"
        % (state_total + work_bytes, (state_total + work_bytes) / 1024.0),
        " */",
    ]

    lines = head + ["", '#include <stddef.h>', '#include "nanomeow.h"', "",
                    "/* 权重池：所有张量的 int8 码与 scale 位流首尾相接（顺序 = 上面那张表），",
                    " * 描述符只存池内 uint16 偏移。scale 位流的口径见 gen_weights.py，与 nanomeow.c",
                    " * 的 nm_row_scale / nm_row_codes 互逆：per-tensor 是 4 B 小端；per-row 是",
                    " * [尾数 23 位][1 B 基线][1 B 增量位宽][增量][行索引]，行索引只在 idx_bits > 0",
                    " * （该张量的码被去重）时才有。model_weights.h 里的原始码与 mul/shift 已经没人",
                    " * 引用，所以本文件不再 include 它（少解析 199 KB）。 */",
                    "const uint8_t nm_pool[%d] = {" % len(pool)]
    for i in range(0, len(pool), 24):
        lines.append("    " + ",".join(str(v) for v in pool[i:i + 24]) + ",")
    lines.append("};")
    lines += ["", "const nm_block nm_blocks[NM_N_LAYER] = {"]
    for layer in range(N_LAYER):
        lines.append("    { /* layer %d */" % layer)
        for field, suffix in BLOCK_FIELDS:
            sym = "nmw_blocks_%d_%s" % (layer, suffix)
            if sym not in lens:
                if field not in LAYER0_ONLY:
                    raise SystemExit("自检失败：第 %d 层缺字段 %s（%s）" % (layer, field, sym))
                lines.append("        %s," % EMPTY_MAT)
                continue
            lines.append("        %s,  /* %s */"
                         % (mat_expr(offs[sym][2:], offs[sym][0], offs[sym][1]), sym))
        lines.append("    },")
    lines.append("};")
    lines.append("")
    for name, sym in TOP_FIELDS:
        lines.append("const nm_mat %s = %s;  /* %s */"
                     % (name, mat_expr(offs[sym][2:], offs[sym][0], offs[sym][1]), sym))
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("写出 %s：%d 个张量，码 %d B + scale %d B = %.1f KiB；RAM 状态 %.1f KiB + 工作区 %.1f KiB"
          % (OUT.name, len(tensors), code_total, scale_total,
             (code_total + scale_total) / 1024.0, state_total / 1024.0, work_bytes / 1024.0))


if __name__ == "__main__":
    main()
