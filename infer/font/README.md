# infer/font — OLED 嵌入式 8x8 紧凑点阵字库

本项目给 nanomeow 上板运行时用的 SSD1306 OLED 字符显示提供一套超轻量的点阵字库。
STM32F103C8T6 只有 64 KiB Flash，空间非常紧张，所以字库按**语料词频筛子集 + 紧凑位流压缩**来做，没收录的字符有统一的兜底渲染机制。

---

## 字形来源与开源许可

点阵字形都取自成熟的开源点阵字体项目（遵循 SIL Open Font License 1.1，许可原件放在 `licenses/`）：

| 字体来源 | 开源协议 | 用途与说明 |
| :--- | :--- | :--- |
| [TakWolf/fusion-pixel-font](https://github.com/TakWolf/fusion-pixel-font) | SIL OFL 1.1 | 8px 等宽泛中日韩点阵字体，字形取自官方发布的 BDF 栅格源文件 |
| [ItMarki/MisekiBitmap](https://github.com/ItMarki/MisekiBitmap) | SIL OFL 1.1 | fusion-pixel-font 简体中文汉字的上游字形来源 |

---

## 选字策略与覆盖率

完整的中日韩 8px 点阵字库有 2.7 万多个字形，体积比 MCU 芯片总容量还大得多。
所以字库用**按 Flash 预算筛选**的策略挑字：

- **ASCII 基础字符**：全部 95 个可打印 ASCII 字符（`0x20` ~ `0x7E`）无条件收全；
- **常用中文字符**：按 SFT 对话语料里 Bot 回复的字符频次从高到低取，在设定的 Flash 预算内尽量多收，最后定了 635 个汉字；
- **总字库容量**：**730 个字符**，对实际训练语料的字符覆盖率是 **98.72%**。

---

## 紧凑位流压缩编码设计

为了在有限空间里塞进更多汉字，字库设计了一套紧凑编码：

1. **有效点阵裁剪**：
   统计发现，这批 8x8 字符按规范摆进格子后，第 0 行和第 7 列永远是空白。所以每个字形实际只存 7 行 × 7 列 = **49 位有效像素**。
2. **紧密位流拼接**：
   49 位数据不做字节对齐，而是首尾相接压进一条连续位流，平均每个字只占 6.125 字节（730 个字一共 4,473 字节）。
3. **LEB128 差分码点索引**：
   把字符的 Unicode 码点按升序排好，只存相邻码点的差值（Delta Encoding），再用 LEB128 变长整数编码，730 字的索引表一共只花 766 字节。
4. **字库总 Flash 开销**：
   点阵位流 (4,473 B) + 码点索引 (766 B) + 兜底字符 (7 B) = **5,246 字节**（算上 C 解码函数总共约 5.6 KB）。

---

## 重新生成字库与构建

如果要按新语料调整收录的字，或者增减预算，跑下面这套工具链：

```powershell
# 1. 运行选字脚本（依据 BDF 源字库与训练语料，按预算生成选字子集清单）
.\.venv\Scripts\python.exe infer\font\select_subset.py

# 2. 运行打包脚本（将 subset_8x8.txt 打包生成 C 语言紧凑位流源文件）
.\.venv\Scripts\python.exe infer\c\tools\gen_font.py
```

*生成产物是 `infer/c/generated/nm_font.c` 和 `infer/c/generated/nm_font.h`。*

---

## 接口与渲染兜底

- **解码 API**：提供 `nm_font_glyph(uint32_t cp, uint8_t out[7])`（按码点解码）和 `nm_font_glyph_utf8(const char *s, int n, uint8_t out[7])`（流式 UTF-8 解码）；
- **兜底渲染保证**：输入里出现字库外的生僻字、非法 UTF-8 序列，或者模型生成的乱码时，解码器统一返回 7x7 像素的空心方框点阵（`nm_font_fallback`），保证前端的显存排版游标能平稳往下走，不会因此中断。