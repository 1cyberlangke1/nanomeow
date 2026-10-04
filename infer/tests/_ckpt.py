"""本地训练产物的定位：三个推理测试共用同一个「当前 checkpoint」来源。

输入：无。输出：checkpoint 路径（可能为 None）。
预期行为：C 引擎用的 `infer/model_weights.h` 是从某个 checkpoint 导出的，Python 参考
          必须读**同一个** checkpoint，否则 G1 的逐位对拍会因为权重不同源而假失败。
          所以优先读头文件里 `export_int8.py` 写下的「权重来源」注释；本地没有那个文件
          （比如训练产物不入库的干净 clone）时，回落到 `train/out/sft_*/sft.pth` 里
          mtime 最新的那个；一个都没有就返回 None（调用方 skip）。
"""

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
WEIGHTS_H = REPO / "infer" / "model_weights.h"
SOURCE_RE = re.compile(r"^/\* 权重来源：(.+) \*/$")


def latest_ckpt():
    """输入：无；输出：与已入库权重同源的 checkpoint 路径；没有则 None。"""
    if WEIGHTS_H.exists():
        for line in WEIGHTS_H.read_text(encoding="utf-8").splitlines():
            m = SOURCE_RE.match(line)
            if m:
                p = pathlib.Path(m.group(1))
                if not p.is_absolute():
                    p = REPO / p
                if p.exists():
                    return p
                break
    cands = sorted(REPO.glob("train/out/sft_*/sft.pth"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None
