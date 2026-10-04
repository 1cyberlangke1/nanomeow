# infer/font — OLED 用的 8x8 子集字库

上板显示模型输出用的点阵字库。**只存子集**：完整的泛中日韩 8px 字库有 27,987 个字形，
整机 Flash 根本放不下（账本见下），所以按语料词频选字，字库外的码点渲染成兜底方框。

## 来源（都是开源点阵字库）

| 项目 | 说明 |
|---|---|
| [TakWolf/fusion-pixel-font](https://github.com/TakWolf/fusion-pixel-font) | 8px 等宽泛中日韩点阵，OFL-1.1，3.2k star。**本字库的字形就取自这里** |
| [ItMarki/MisekiBitmap](https://github.com/ItMarki/MisekiBitmap) | 8x8 简体汉字字形来源（fusion-pixel 的 8px 简体就是它），OFL-1.1 |
| [dhepper/font8x8](https://github.com/dhepper/font8x8) | 纯 ASCII 8x8，公有领域；**本字库没用它**，留作对照 |

字形取自 fusion-pixel-font 官方 Release 的 `fusion-pixel-font-8px-monospaced-bdf-v2026.02.27.zip`
里的 `latin` 与 `zh_hans` 两份 BDF。BDF 是官方栅格化的结果，逐位确定，比拿 PIL 现渲染可靠。
许可原件在 `licenses/`。

## 重新生成

BDF 单份 3 MB，不入库（`tmp/` 已被 .gitignore 忽略）。要重新选字就先下载解压：

```powershell
$env:HTTPS_PROXY='http://127.0.0.1:7890'
curl.exe -sL -o tmp\f8.bdf.zip https://github.com/TakWolf/fusion-pixel-font/releases/download/2026.02.27/fusion-pixel-font-8px-monospaced-bdf-v2026.02.27.zip
.\.venv\Scripts\python.exe -c "import zipfile; zipfile.ZipFile(r'tmp\f8.bdf.zip').extractall(r'tmp\f8bdf')"
```

然后两步（都带自检，不一致就报错退出）：

```powershell
.\.venv\Scripts\python.exe infer\font\select_subset.py   # BDF + 语料 -> subset_8x8.txt
.\.venv\Scripts\python.exe infer\c\gen_font.py           # subset_8x8.txt -> infer/c/nm_font.c / .h
```

`select_subset.py` 需要 BDF 与语料，`gen_font.py` 只需要 `subset_8x8.txt`；
所以干净 clone（没有 BDF 也没有语料）也能重新生成 C 字库。改容量只动 `--budget` 一个参数
（单位是打包后点阵 + 码点表的字节数）。

## 存法

- 8x8 格里，实测这批字形**第 0 行与第 7 列恒空**，所以每字只存 7 行 x 7 列 = **49 位**
  （行内 bit7..bit1 是第 0..6 列）。摆格后只要哪一行越界或第 7 列非空，选字脚本直接报错。
- 这 49 位**紧密打包成位流**，不是每行 1 字节 —— 每字 6.125 B。表尾补 1 个 0 字节当哨兵，
  因为解包固定读「当前字节 + 下一个字节」（`nm_font_read7` 用 16 位窗口右移取 7 位）。
- 码点表按升序存成「首个绝对 + 之后增量」的 LEB128 流，701 字只花 736 B。
- 查表是顺序扫描 + 提前退出（码点升序，超过目标就停），字库外与非法 UTF-8 都写兜底字形
  （7x7 空心方框）。接口是「解包到调用方给的 7 字节缓冲」：`nm_font_glyph(cp, out)` /
  `nm_font_glyph_utf8(s, n, out)`，OLED 驱动拿到一个完整字符后调后者即可。

## 容量账本（实测，Cortex-M3 clang -Oz）

| 项 | 字节 |
|---|---|
| 点阵位流 4,295 + 码点表 736 + 兜底 7 | 5,038 |
| 查表代码（`nm_font_glyph` / `nm_font_glyph_utf8`，含内联的 `nm_font_read7`） | 380 |
| **字库合计** | **5,418** |

整机 Flash **65,490 B**，64 KiB 余 **46 B**；RAM 不变（字库全在 Flash）。
账本由 `infer/firmware/build_firmware.py` 真链接量出，不是估算。

## 为什么是全量 701 字

语料 bot 侧共 701 个不同字符，现在**全部收录** —— `select_subset.py` 的 `--budget` 按打包后的
实际占用算（`packed_size(n) + varint_size(cps)`），不是按 7 字节/字的原样。

| 方案 | 点阵 + 码点表 | 语料字符覆盖 | 整条回复可完整渲染 |
|---|---|---|---|
| 频率前 357 字（budget 2,900，未打包） | 2,899 B | 89.98% | 25.12% |
| 频率前 422 字（budget 3,412，未打包） | 3,412 B | 92.67% | 40.63% |
| **全量 701 字（当前，打包后）** | **5,031 B** | **100%** | **100%（语料内）** |

模型真实输出侧（38 条演示/泛化问答共 735 字符）：**演示 20 条零方框**，全体还剩 **17 个方框（2.31%）**，
全是模型跑飞时吐的**语料外**字符（`✉䜈丆享兌含坸戩攀槠聜裉蹸躬辜酌龼`）—— 全量字库也覆盖不到，
所以兜底字形无论如何都要有。

之前要全量还差 2,206 B，靠这几步**无损**优化腾出来（每步都单独提交、逐位可验）：

| 优化 | 省 |
|---|---|
| 描述符 `nm_mat` 的指针改统一权重池 `nm_pool` 的 uint16 偏移（12 B → 8 B） | 428 B |
| 字库点阵 7 字节/字 → 49 位/字紧密打包 | 368 B |
| exp / log1p 两张 Q15 表改二阶差分位流，运行期纯整数展开 | 772 B |
| 上板 CFLAGS 补 `-ffunction-sections -fdata-sections`，`--gc-sections` 不再空转 | 256 B |

权重侧已经榨干：int8 码的字节熵实测 7.722 bit（均匀是 8.0），无损压缩上限只有 3.5%，
而且要在内层 GEMV 循环里解压，得不偿失。`.ARM.exidx` 那 496 B 已经落地（见 `firmware/m3.ld`）。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest infer\tests\test_font.py -q
```

- 生成的 C 字库 == `subset_8x8.txt` == BDF 原件，逐位一致（BDF 不在时那条 skip）；
  C 里的位流会被反解回 7 字节/字再逐位比。
- `infer/c/nm_font_selftest.c`：码点表严格升序、每个字形都能解包、解包后按同一口径**重新打包**
  必须与位流逐位相同（重新打包那份用逐位写入独立写一遍，不是自洽比较）、字库外与非法 UTF-8 给兜底。