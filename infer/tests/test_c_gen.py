"""生成路径闸门：C 引擎的贪心生成与 Python 定点参考逐位一致，且只吐完整字符。

输入：gcc 编译 infer/c/ 下的引擎（nm_chat.c / nm_gen.c / nanomeow.c / nm_weights.c，
      以及解码器对拍入口 nm_utf8_selftest.c）。
输出：pytest 断言 —— 同一 prompt / 同一参数下，C 的生成字节流与 Python 参考完全相同；
      生成结果严格 UTF-8 解码不抛错（没有半截序列、没有 U+FFFD）；
      解码器在「半截 / 非法 / 过长 / 代理区 / 越界」字节流上与 Python 参考输出与计数一致。
预期行为：本机没有 gcc 时 skip；权重头与 checkpoint 必须同源。
"""

import pathlib
import random
import shutil
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(REPO / "train"))

from _ckpt import latest_ckpt  # noqa: E402
from ref.model import Int8Model  # noqa: E402
from ref.weights import load_weights  # noqa: E402
from src.generate import build_prompt  # noqa: E402
from src.tokenizer import ETX_ID, UTF8StreamDecoder, encode  # noqa: E402

C_DIR = HERE.parents[1] / "c"
# 头文件分散在 engine / generated / display / platform 四个子目录，编译时一起加 -I
C_INCLUDES = [a for p in ("", "engine", "generated", "display", "platform")
              for a in ("-I", str(C_DIR / p))]
CKPT = latest_ckpt()
GCC = shutil.which("gcc")
Q16 = 65536
PEN_13 = 85197          # round(1.3 * 2^16)，与 Python 侧用同一个整数


def _compile(tmp_path, name, sources):
    """输入：临时目录、可执行名、源文件列表；输出：编译好的路径。预期行为：-Werror 一次编过。"""
    exe = tmp_path / name
    subprocess.run(
        [GCC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", *C_INCLUDES,
         "-o", str(exe)] + [str(s) for s in sources],
        check=True, capture_output=True, text=True)
    return exe


@pytest.fixture(scope="module")
def chat_exe(tmp_path_factory):
    """输入：无；输出：主机端生成 CLI（prompt 走 stdin，生成字节走 stdout）。"""
    return _compile(tmp_path_factory.mktemp("c_gen"), "nm_chat.exe",
                    [C_DIR / "host" / "nm_chat.c", C_DIR / "engine" / "nm_gen.c",
                     C_DIR / "engine" / "nanomeow.c", C_DIR / "generated" / "nm_weights.c"])


@pytest.fixture(scope="module")
def utf8_exe(tmp_path_factory):
    """输入：无；输出：增量 UTF-8 解码器的对拍 CLI。"""
    return _compile(tmp_path_factory.mktemp("c_utf8"), "nm_utf8.exe",
                    [C_DIR / "selftest" / "nm_utf8_selftest.c"])


def _bytes(text):
    """输入：str；输出：UTF-8 字节串。预期行为：把 tokenizer.encode 的 int 列表变成 bytes。"""
    return bytes(encode(text))


def _key(code, seen, pen_q16):
    """输入：候选码、是否已生成过、惩罚系数 Q16；输出：精确整数 key（越大越优先）。

    预期行为：与 C 的 nm_gen_key 逐字对应 —— 把「没出现过」「出现且正码」「出现且负码」
              三个式子通分到同一个分母后比 key。
    """
    if pen_q16 <= Q16:
        return code * Q16 * Q16
    if not seen:
        return code * pen_q16 * Q16
    if code > 0:
        return code * Q16 * Q16
    if code < 0:
        return code * pen_q16 * pen_q16
    return 0


def python_generate(model, prompt_bytes, max_new_tokens, pen_q16=Q16, window=0, cap=512):
    """输入：Int8Model、prompt 字节、生成参数；输出：(文本, 是否命中 <ETX>)。

    预期行为：与 C 的 nm_generate 同一套整数算法 —— 贪心、并列取最小下标、
              <ETX> 停止、生成字节过增量 UTF-8 解码。
    """
    states = model.zeros_state()
    if not prompt_bytes:
        logits = model.forward_token(0, states)
    else:
        for byte in prompt_bytes:
            logits = model.forward_token(byte, states)
    dec = UTF8StreamDecoder()
    text = ""
    hist, count = [], {}
    hit = False
    for _ in range(max_new_tokens):
        best, best_key = 0, _key(logits.codes[0], count.get(0, 0) > 0, pen_q16)
        for i, code in enumerate(logits.codes):
            k = _key(code, count.get(i, 0) > 0, pen_q16)
            if k > best_key:
                best_key, best = k, i
        if best == ETX_ID:
            hit = True
            break
        text += dec.push(best)
        hist.append(best)
        count[best] = count.get(best, 0) + 1
        if len(hist) > cap:
            old = hist.pop(0)
            count[old] -= 1
        logits = model.forward_token(best, states)
    dec.flush()
    return text, hit


def run_chat(exe, prompt_bytes, max_new_tokens, pen_q16=Q16, window=0):
    """输入：CLI、prompt 字节、生成参数；输出：(生成文本, 定点越界哨兵, 是否命中 <ETX>)。

    预期行为：stdout 严格按 UTF-8 解码（解码失败就说明吐了半截序列）；
              哨兵与命中标记从 stderr 的 `R <n>` / `H <n>` 读。
    """
    proc = subprocess.run([str(exe), str(max_new_tokens), str(pen_q16), str(window)],
                          input=prompt_bytes, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    parts = proc.stderr.decode("ascii").split()
    assert parts[0] == "R" and parts[2] == "H", proc.stderr
    return proc.stdout.decode("utf-8"), int(parts[1]), int(parts[3])


@pytest.mark.skipif(GCC is None, reason="需要 gcc 才能编译 C 引擎")
@pytest.mark.skipif(CKPT is None, reason="需要训练产出的 checkpoint")
def test_c_generate_matches_python_bitwise(chat_exe):
    """逐位对拍：同一 prompt 下 C 与 Python 参考生成的文本完全相同，且没有定点越界。"""
    cfg, wts = load_weights(str(CKPT))
    model = Int8Model(cfg, wts)
    prompt = _bytes(build_prompt("你好"))
    want_text, want_hit = python_generate(model, prompt, 48, PEN_13, 0)
    got_text, range_error, hit = run_chat(chat_exe, prompt, 48, PEN_13, 0)
    assert range_error == 0
    assert got_text == want_text
    assert hit == want_hit
    assert want_text, "参考生成不能是空串"


@pytest.mark.skipif(GCC is None, reason="需要 gcc 才能编译 C 引擎")
@pytest.mark.skipif(CKPT is None, reason="需要训练产出的 checkpoint")
def test_c_generate_matches_python_with_window(chat_exe):
    """换一组参数（关惩罚 + 滑动窗口）再对拍一次，覆盖历史环形的淘汰路径。"""
    cfg, wts = load_weights(str(CKPT))
    model = Int8Model(cfg, wts)
    prompt = _bytes(build_prompt("再见"))
    want_text, _ = python_generate(model, prompt, 32, PEN_13, 8, cap=8)
    got_text, range_error, _ = run_chat(chat_exe, prompt, 32, PEN_13, 8)
    assert range_error == 0
    assert got_text == want_text


@pytest.mark.skipif(GCC is None, reason="需要 gcc 才能编译 C 引擎")
@pytest.mark.skipif(CKPT is None, reason="需要训练产出的 checkpoint")
def test_c_generate_no_mojibake(chat_exe):
    """不乱码：严格 UTF-8 解码通过、不含 U+FFFD，而且真的吐出了中文。"""
    prompt = _bytes(build_prompt("今天怎么样"))
    text, range_error, _ = run_chat(chat_exe, prompt, 64, Q16, 0)
    assert range_error == 0
    assert "\ufffd" not in text
    assert text
    assert any("\u4e00" <= ch <= "\u9fff" for ch in text), repr(text)


DECODER_CASES = [
    b"",
    b"hello",
    _bytes("你好，喵～"),
    b"\xe4",                    # 半截 3 字节序列
    b"\xe4\xbd",
    b"\xe4A",                   # 续字节非法
    b"\xe4\xbdA",
    b"\xf0\x9f\x98\x80",        # 4 字节 emoji
    b"\xf0\x9f\x98",            # 半截 emoji
    b"\xff\xfe",                # 非法首字节
    b"\xc0\x80",                # 过长编码
    b"\xe0\x80",                # 过长编码的半截
    b"\xe0\x80\x80",
    b"\xf0\x80\x80\x80",
    b"\xed\xa0\x80",            # 代理区
    b"\xed\xa0",
    b"\xf4\x90\x80\x80",        # 超出 U+10FFFF
    b"\xf5\x80\x80\x80",
    b"A\xffB",
    b"\x80\x80",
]


@pytest.mark.skipif(GCC is None, reason="需要 gcc 才能编译 C 引擎")
def test_c_utf8_decoder_matches_python(utf8_exe):
    """解码器对拍：任意字节流的输出字节、丢弃数与挂起数都与 Python 参考一致。"""
    rng = random.Random(20261004)
    cases = list(DECODER_CASES)
    cases.append(bytes(rng.randrange(256) for _ in range(400)))
    cases.append(bytes(rng.randrange(0x80, 0xF8) for _ in range(200)))
    for data in cases:
        proc = subprocess.run([str(utf8_exe)], input=data, capture_output=True)
        assert proc.returncode == 0, proc.stderr
        dec = UTF8StreamDecoder()
        want = dec.push_many(data)
        assert proc.stdout.decode("utf-8") == want, repr(data)
        parts = proc.stderr.decode("ascii").split()
        assert int(parts[1]) == dec.dropped_bytes, repr(data)
        assert int(parts[3]) == dec.pending_bytes, repr(data)
