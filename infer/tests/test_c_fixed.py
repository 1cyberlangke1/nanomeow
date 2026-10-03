"""G1 底座：nm_fixed.h 与 infer/ref/fixed.py 的逐位对拍。

输入：随机用例（scale / 整数 / 取码式）。
输出：pytest 断言；C 侧结果必须与 Python 参考逐位相同。
预期行为：本机没有 gcc 时 skip，其余情况必须全绿 —— 这是 G1「Python 定点参考 == C 引擎」
          的第一层（定点底座），模型层的对拍在 test_c_engine.py。
"""

import pathlib
import random
import shutil
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from ref.fixed import (  # noqa: E402
    apply_scale,
    div,
    div_int,
    mul,
    mul_int,
    rescale,
    round_div,
    scale_from_frac,
)

C_DIR = HERE.parents[1] / "c"
GCC = shutil.which("gcc")


def _build(tmp_path):
    """输入：临时目录；输出：编译好的自测程序路径。预期行为：gcc 一次编过，无警告。"""
    exe = tmp_path / "nm_fixed_selftest.exe"
    subprocess.run(
        [GCC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
         "-o", str(exe), str(C_DIR / "nm_fixed_selftest.c")],
        check=True, capture_output=True, text=True)
    return exe


def _run(exe, cases):
    """输入：程序路径、用例列表（每项是一条指令的字符串）；输出：C 侧输出行列表。"""
    proc = subprocess.run([str(exe)], input="\n".join(cases) + "\n",
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()


def _rand_scale(rng):
    """输入：随机源；输出：归一化 scale（m ∈ [2^30, 2^31)、e ∈ [-40, 10]）。"""
    return (rng.randrange(1 << 30, 1 << 31), rng.randrange(-40, 11))


@pytest.mark.skipif(GCC is None, reason="需要 gcc 才能编译 C 引擎")
def test_fixed_layer_matches_python(tmp_path):
    """定点底座全算子对拍：round_div / mul / div / apply_scale / mul_int / div_int /
    rescale / requant 取码 / scale_from_frac。"""
    rng = random.Random(20261004)
    cases, expect = [], []

    for _ in range(200):
        num = rng.randrange(-(1 << 62), 1 << 62)
        den = rng.randrange(1, 1 << 20)
        cases.append("rd %d %d" % (num, den))
        expect.append("%d" % round_div(num, den))

    for _ in range(200):
        a, b = _rand_scale(rng), _rand_scale(rng)
        cases.append("nm %d %d %d %d" % (a[0], a[1], b[0], b[1]))
        expect.append("%d %d" % mul(a, b))
        cases.append("nd %d %d %d %d" % (a[0], a[1], b[0], b[1]))
        expect.append("%d %d" % div(a, b))

    for _ in range(200):
        s = _rand_scale(rng)
        val = rng.randrange(-(1 << 20), 1 << 20)
        cases.append("nas %d %d %d" % (val, s[0], s[1]))
        expect.append("%d" % apply_scale(val, s))
        k = rng.randrange(1, 1 << 24)
        cases.append("nmi %d %d %d" % (s[0], s[1], k))
        expect.append("%d %d" % mul_int(s, k))
        cases.append("ndi %d %d %d" % (s[0], s[1], k))
        expect.append("%d %d" % div_int(s, k))

    # rescale 用现实指数域：本模型的 scale 指数都远小于 -frac_bits，
    # 也就是「右移」那条路；左移只在极端 scale 下出现，由 nm_shl_checked 兜底。
    for _ in range(200):
        s = (rng.randrange(1 << 30, 1 << 31), rng.randrange(-60, -16))
        val = rng.randrange(-(1 << 24), 1 << 24)
        cases.append("nrs %d %d %d 16" % (val, s[0], s[1]))
        expect.append("%d" % rescale(val, s, 16))

    # 取码式：num ≥ 127 * max|v|（引擎里的实际口径），商落在 [-128, 128]
    for _ in range(200):
        v = rng.randrange(-(1 << 51), 1 << 51)
        num = abs(v) * 127 + rng.randrange(0, 1 << 20)
        cases.append("nrc %d %d" % (v, num))
        # 期望值就是取码式本身：round(v * 16256 / num)，与 quantize_dynamic 里的那一行同式
        expect.append("%d" % round_div(v * 16256, num))

    for f in range(0, 33):
        cases.append("nsf %d" % f)
        expect.append("%d %d" % scale_from_frac(f))

    got = _run(_build(tmp_path), cases)
    assert len(got) == len(expect)
    for i, (g, e) in enumerate(zip(got, expect)):
        assert g == e, "第 %d 条不一致：C=%s Python=%s（%s）" % (i, g, e, cases[i])