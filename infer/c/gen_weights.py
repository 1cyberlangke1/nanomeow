"""从 infer/model_weights.h 生成 C 引擎的权重描述表 infer/c/nm_weights.c。

输入：infer/model_weights.h（由 train/scripts/export_int8.py 生成）、infer/c/nanomeow.h 与
      infer/c/nanomeow.c（用来统计引擎的运行期内存）。
输出：infer/c/nm_weights.c（`nm_blocks[3]` 与四个顶层 `nm_mat` 的定义）。
预期行为：把「int8 码 + 每行 int32 乘子 + 每行 int8 移位」装成 `nm_mat` ——
          头文件里 `<符号>_mul` 是数组的是 per-row（rows = 乘子条数、cols = 码数 / rows），
          是标量的是 per-tensor（rows = 1、cols = 码数）。
          文件头的注释由实际参数统计出来：每个导出权重的名字与字节数、权重合计、
          Flash 常驻表大小、RAM（状态 + 引擎工作区）大小，全部现算，不写死。
          生成前自检：3 层每个字段都找得到、码数能被行数整除，否则报错退出。
"""

import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
HEADER = HERE.parent / "model_weights.h"
ENGINE_H = HERE / "nanomeow.h"
ENGINE_C = HERE / "nanomeow.c"
OUT = HERE / "nm_weights.c"
CFG = HERE.parent / "model_cfg.h"    # 结构常量的唯一来源，由导出脚本按 checkpoint 生成
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

EMPTY_MAT = "{NULL, NULL, NULL, 0, 0, 0}"
SCALE_BYTES = 5          # 一条 scale = int32 乘子 + int8 移位


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


def mat_expr(sym, lens, is_array):
    """输入：符号名、元素个数表、数组标记表；输出：一个 nm_mat 初始化表达式。"""
    if sym not in lens:
        raise SystemExit("自检失败：头文件里没有符号 %s" % sym)
    n = lens[sym]
    if is_array.get(sym + "_mul", False):
        rows = lens[sym + "_mul"]
        if rows <= 0 or n % rows:
            raise SystemExit("自检失败：%s 的码数 %d 不是行数 %d 的整数倍" % (sym, n, rows))
        return "{%s, %s_mul, %s_shift, %d, %d, 1}" % (sym, sym, sym, rows, n // rows)
    return "{%s, &%s_mul, &%s_shift, 1, %d, 0}" % (sym, sym, sym, n)


def parse_defines(text):
    """输入：infer/model_cfg.h 文本；输出：{NM_X: 整数}。

    预期行为：只认 `#define NMW_X <十进制>`，返回时把 NMW_ 前缀换成 NM_。结构常量由
              train/scripts/export_int8.py 按 checkpoint 生成，这里跟着它走，不再抄一份。
    """
    return {"NM_" + m.group(1): int(m.group(2))
            for m in re.finditer(r"^#define NMW_(\w+) (\d+)$", text, re.M)}


def state_bytes(d):
    """输入：宏表；输出：单层 nm_layer_state 的字节数。

    预期行为：镜像 infer/c/nanomeow.h 里 nm_layer_state 的布局 ——
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


def main():
    """输入：无；输出：写 nm_weights.c 并在 stdout 打印一行摘要。"""
    lens, is_array, order = parse_header(HEADER.read_text(encoding="utf-8"))
    d = parse_defines(CFG.read_text(encoding="utf-8"))
    work_bytes, work_n = workspace_bytes(ENGINE_C.read_text(encoding="utf-8"), d)

    tensors = [s for s in order if is_tensor(s, lens)]
    code_total = sum(lens[s] for s in tensors)
    scale_total = sum(lens[s + "_mul"] for s in tensors) * SCALE_BYTES
    state_total = state_bytes(d) * d["NM_N_LAYER"]

    head = ["/* 自动生成，请勿手改：infer/c/gen_weights.py",
            " *",
            " * 权重来源：infer/model_weights.h（train/scripts/export_int8.py 从 checkpoint 导出）",
            " * 模型：%d 层 RWKV-7 x070 缩维，n_embd=%d，n_head=%d，head_size=%d，"
            "dim_ffn=%d，词表 %d"
            % (d["NM_N_LAYER"], d["NM_N_EMBD"], d["NM_N_HEAD"], d["NM_HEAD_SIZE"],
               d["NM_DIM_FFN"], d["NM_VOCAB"]),
            " *",
            " * 导出的权重（名字 = 训练侧参数名；字节数 = int8 码 / scale 表）："]
    for sym in tensors:
        if is_array.get(sym + "_mul", False):
            rows = lens[sym + "_mul"]
            kind = "per-row %d x %d" % (rows, lens[sym] // rows)
            scale_n = rows
        else:
            kind = "per-tensor %d" % lens[sym]
            scale_n = 1
        head.append(" *   %-32s %6d B / %5d B  (%s)"
                    % (display_name(sym), lens[sym], scale_n * SCALE_BYTES, kind))
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

    lines = head + ["", '#include "../model_weights.h"', '#include "nanomeow.h"', "",
                    "const nm_block nm_blocks[NM_N_LAYER] = {"]
    for layer in range(N_LAYER):
        lines.append("    { /* layer %d */" % layer)
        for field, suffix in BLOCK_FIELDS:
            sym = "nmw_blocks_%d_%s" % (layer, suffix)
            if sym not in lens:
                if field not in LAYER0_ONLY:
                    raise SystemExit("自检失败：第 %d 层缺字段 %s（%s）" % (layer, field, sym))
                lines.append("        %s," % EMPTY_MAT)
                continue
            lines.append("        %s," % mat_expr(sym, lens, is_array))
        lines.append("    },")
    lines.append("};")
    lines.append("")
    for name, sym in TOP_FIELDS:
        lines.append("const nm_mat %s = %s;" % (name, mat_expr(sym, lens, is_array)))
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("写出 %s：%d 个张量，码 %d B + scale %d B = %.1f KiB；RAM 状态 %.1f KiB + 工作区 %.1f KiB"
          % (OUT.name, len(tensors), code_total, scale_total,
             (code_total + scale_total) / 1024.0, state_total / 1024.0, work_bytes / 1024.0))


if __name__ == "__main__":
    main()
