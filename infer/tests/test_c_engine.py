"""G1 闸门：C 引擎与 Python 定点参考的逐位对拍。

输入：gcc 编译 infer/c/ 下的引擎（nm_engine_selftest.c + nanomeow.c + nm_weights.c）。
输出：pytest 断言 —— 同一串 token 下每一步的 logits 码与 scale 必须逐位相同。
预期行为：本机没有 gcc 时 skip；权重头与 checkpoint 必须同源（否则第一步就对不上）。
"""

import json
import pathlib
import shutil
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(REPO / "train"))

from ref.model import Int8Model  # noqa: E402
from ref.weights import load_weights  # noqa: E402

C_DIR = HERE.parents[1] / "c"
CKPT = REPO / "train" / "out" / "sft_dyn_fine" / "sft_step2750.pth"
DATASET = REPO / "train" / "dataset" / "nana_clean.jsonl"
GCC = shutil.which("gcc")
# 一段真实的对话字节（user:你好\nbot: 的开头），覆盖多字节 UTF-8 与 ASCII
TOKENS = [0x75, 0x73, 0x65, 0x72, 0x3A, 0xE4, 0xBD, 0xA0, 0xE5, 0xA5, 0xBD, 0x0A,
          0x62, 0x6F, 0x74, 0x3A, 0xE6, 0x97, 0xA9, 0xE4, 0xB8, 0x8A, 0xE5, 0xA5, 0xBD]


@pytest.fixture(scope="module")
def engine_exe(tmp_path_factory):
    """输入：无；输出：编译好的 C 引擎路径。预期行为：gcc 一次编过，无警告（-Werror）。"""
    exe = tmp_path_factory.mktemp("c_engine") / "nm_engine.exe"
    subprocess.run(
        [GCC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(C_DIR),
         "-o", str(exe), str(C_DIR / "nm_engine_selftest.c"), str(C_DIR / "nanomeow.c"),
         str(C_DIR / "nm_weights.c")],
        check=True, capture_output=True, text=True)
    return exe


def run_engine(exe, tokens):
    """输入：引擎路径、token 串；输出：(每一步的 (码, scale), 定点溢出哨兵)。"""
    proc = subprocess.run([str(exe)], input="%d\n%s\n" % (len(tokens), " ".join(map(str, tokens))),
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    steps, range_error, pending = [], None, None
    for line in proc.stdout.splitlines():
        if line.startswith("S "):
            _, m, e = line.split()
            pending = (int(m), int(e))
        elif line.startswith("R "):
            range_error = int(line.split()[1])
        elif pending is not None:
            steps.append(([int(v) for v in line.split()], pending))
            pending = None
    return steps, range_error


@pytest.mark.skipif(GCC is None, reason="需要 gcc 才能编译 C 引擎")
@pytest.mark.skipif(not CKPT.exists(), reason="需要训练产出的 checkpoint")
def test_c_engine_matches_python_bitwise(engine_exe):
    """逐位对拍：每一步的 logits 码与 scale 与 Python 参考完全一致，且没有触发溢出哨兵。"""
    steps, range_error = run_engine(engine_exe, TOKENS)
    assert len(steps) == len(TOKENS)
    assert range_error == 0

    cfg, wts = load_weights(str(CKPT))
    eng = Int8Model(cfg, wts)
    state = eng.zeros_state()
    for i, token in enumerate(TOKENS):
        out = eng.forward_token(token, state)
        codes, scale = steps[i]
        assert scale == out.scale, "第 %d 步 scale 不一致：C=%s Python=%s" % (i, scale, out.scale)
        assert codes == out.codes, "第 %d 步 logits 码不一致" % i


def long_tokens():
    """输入：无；输出：512 字节真实语料的 token 串。预期行为：取清洗语料首行，不足就自拼。"""
    with DATASET.open(encoding="utf-8") as f:
        text = json.loads(f.readline())["text"]
    data = text.encode("utf-8")
    while len(data) < 512:
        data = data + data
    return list(data[:512])


@pytest.mark.skipif(GCC is None, reason="需要 gcc 才能编译 C 引擎")
@pytest.mark.skipif(not CKPT.exists(), reason="需要训练产出的 checkpoint")
@pytest.mark.skipif(not DATASET.exists(), reason="需要清洗后的语料")
def test_c_engine_long_context_bitwise(engine_exe):
    """512 token 长上下文的逐位对拍。

    预期行为：每一步的 logits 码与 scale 都与 Python 参考相同、哨兵为 0。
              这条用例覆盖「中间码乘积超过 int64」的长程累积场景 —— 25 token 的短 prompt
              在第 179 步之前就结束了，抓不到它。
    """
    tokens = long_tokens()
    steps, range_error = run_engine(engine_exe, tokens)
    assert len(steps) == len(tokens)
    assert range_error == 0

    cfg, wts = load_weights(str(CKPT))
    eng = Int8Model(cfg, wts)
    state = eng.zeros_state()
    for i, token in enumerate(tokens):
        out = eng.forward_token(token, state)
        codes, scale = steps[i]
        assert scale == out.scale, "第 %d 步 scale 不一致：C=%s Python=%s" % (i, scale, out.scale)
        assert codes == out.codes, "第 %d 步 logits 码不一致" % i
