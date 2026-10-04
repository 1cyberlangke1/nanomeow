# -*- coding: utf-8 -*-
"""真机闸门：上板固件在 Cortex-M3 模拟器（Unicorn）里真的跑起来。

输入：`build_firmware.py` 真交叉编译 + 真链接出的固件 ELF，以及 gcc 编出来的主机 C 引擎。
输出：pytest 断言 —— 固件从 USART1 吐出的字节，必须与同一 prompt 下主机引擎的输出逐字节相同
      （固件生成完会多打一个换行，那是它自己的收尾，不算差异）；输出必须是完整 UTF-8。
预期行为：没有 unicorn 或没有 ARM 工具链时 skip。这是「编译通过」之外的「跑起来」证据 ——
          跑的是真机指令，不是行为模型；顺带守住复位向量第 0 项必须是栈顶数值（不是它的地址）。
          文件末尾还把同一次模拟里固件发出去的**软件 I2C 位流**还原成 OLED 屏幕，与字库对拍。
"""

import pathlib
import shutil
import struct
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[2]
INFER = HERE.parents[1]
FW_DIR = INFER / "firmware"
C_DIR = INFER / "c"
# 头文件分散在 engine / generated / display / platform 四个子目录，编译时一起加 -I
C_INCLUDES = [a for p in ("", "engine", "generated", "display", "platform")
              for a in ("-I", str(C_DIR / p))]

pytest.importorskip("unicorn", reason="需要 pip install unicorn 才能跑 M3 模拟闸门")

sys.path.insert(0, str(FW_DIR))
import run_m3  # noqa: E402

sys.path.insert(0, str(C_DIR / "tools"))
import gen_font  # noqa: E402
import oled_view  # noqa: E402

GCC = shutil.which("gcc")
LINE = "你好"                                  # 固件自己会拼成 user:你好\nbot:
PROMPT = "user:你好\nbot:"
MAX_NEW, PEN_Q16, WINDOW = "128", "65536", "0"  # 与 keil_demo/User/main.c 的 g_cfg 同口径


@pytest.fixture(scope="module")
def fw_elf(tmp_path_factory):
    """输入：无；输出：真交叉编译 + 真链接出来的固件 ELF。

    预期行为：直接调仓库自己的上板构建脚本；clang / zig 缺一个就 skip，不假装跑过。
    """
    out = tmp_path_factory.mktemp("m3_fw")
    rc = subprocess.run([sys.executable, str(FW_DIR / "build_firmware.py"),
                         "--build-dir", str(out)], capture_output=True, text=True)
    if rc.returncode:
        pytest.skip("上板工具链不可用：%s" % rc.stdout.strip().splitlines()[-1:])
    return out / "nanomeow.elf"


@pytest.fixture(scope="module")
def chat_exe(tmp_path_factory):
    """输入：无；输出：主机端生成 CLI（prompt 走 stdin，生成字节走 stdout）。"""
    if GCC is None:
        pytest.skip("需要 gcc 才能编主机 C 引擎")
    exe = tmp_path_factory.mktemp("m3_chat") / "nm_chat.exe"
    subprocess.run(
        [GCC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", *C_INCLUDES,
         "-o", str(exe), str(C_DIR / "host" / "nm_chat.c"), str(C_DIR / "engine" / "nm_gen.c"),
         str(C_DIR / "engine" / "nanomeow.c"), str(C_DIR / "generated" / "nm_weights.c")],
        check=True, capture_output=True, text=True)
    return exe


def test_firmware_runs_on_cortex_m3(fw_elf, chat_exe):
    """固件在 Cortex-M3 上跑完一整轮「读一行 → 生成 → 回到等待」，输出与主机引擎逐字节一致。"""
    host = subprocess.run([str(chat_exe), MAX_NEW, PEN_Q16, WINDOW],
                          input=PROMPT.encode("utf-8"), capture_output=True)
    assert host.returncode == 0, host.stderr
    m3, _ = run_m3.run_firmware(fw_elf, LINE.encode("utf-8") + b"\n")
    assert m3, "固件一个字都没吐出来"
    assert m3 == host.stdout + b"\n", "固件输出与主机引擎不一致：%r vs %r" % (m3, host.stdout)
    assert b"\xef\xbf\xbd" not in m3, "输出里有 U+FFFD，说明吐了半截 UTF-8"
    m3.decode("utf-8")                          # 严格解码，半截序列会直接抛错


def test_reset_vector_holds_stack_top(fw_elf):
    """复位向量第 0 项必须是栈顶**数值**且落在 RAM 里 —— 写成 `&变量`（Flash 地址）真机第一次压栈就硬 fault。"""
    blob = dict(run_m3.load_elf(fw_elf))[run_m3.FLASH_BASE]
    sp = struct.unpack_from("<I", blob, 0)[0]
    assert run_m3.RAM_BASE <= sp <= run_m3.RAM_BASE + run_m3.RAM_SIZE, \
        "栈顶 0x%08x 不在 RAM 里，复位向量写错了" % sp


# --- OLED 可视化：把固件真发出去的软件 I2C 位流还原成屏幕，再和字库对拍 ---

# nm_font.c 的兜底字形（7x7 空心方框），字库里没有的码点用它；gen_font.py 里是同一份数据。
FALLBACK = gen_font.FALLBACK

# 与 infer/c/display/nm_oled.h 的 NM_OLED_SCALE 同口径：字库是 8x8，上屏时每个像素铺成 SCALE x SCALE。
OLED_SCALE = 2
OLED_CELL = 8 * OLED_SCALE                      # 一个字的像素边长
OLED_COLS = oled_view.OLED_W // OLED_CELL       # 一行几个字
OLED_LINE_PAGES = OLED_SCALE                    # 一行字占几页


@pytest.fixture(scope="module")
def font_glyphs():
    """输入：无；输出：{码点: 7 行点阵}。

    预期行为：读 infer/font/subset_8x8.txt —— 它就是 gen_font.py 生成 nm_font.c 的那份输入
              （gen_font 自己会自检两边一致），所以这里算出来的字形就是固件真用的字形。
    """
    return dict(gen_font.read_subset(gen_font.SUBSET))


@pytest.fixture(scope="module")
def oled_run(fw_elf):
    """输入：无；输出：(串口字节, OledBus)。

    预期行为：跑一轮真固件，OLED 位流与串口输出一起收下来 —— 下面三个闸门共用这一次模拟，
              不重复跑（一次模拟要几秒）。
    """
    return oled_view.run_with_oled(fw_elf, LINE.encode("utf-8") + b"\n")


def render_line(text, glyphs, col=0):
    """输入：一行文字、字形表、起始格号；输出：从页 0 起、OLED_LINE_PAGES 页拼成的字节串。

    预期行为：照抄 nm_oled_blit 的放大规则 —— 字库第 r 行第 c 列的点铺成 SCALE x SCALE 实心方块，
              落在字块相对像素 (r*SCALE, c*SCALE)；第 i 个字从第 col+i 格起，一个字占 OLED_CELL
              列，字块按页组织；字库外的字符用兜底方框。
    """
    fb = bytearray(OLED_LINE_PAGES * oled_view.OLED_W)
    for i, ch in enumerate(text):
        blob = glyphs.get(ord(ch), FALLBACK)
        for r in range(7):
            for c in range(8):
                if blob[r] & (0x80 >> c):
                    for dy in range(OLED_SCALE):
                        y = r * OLED_SCALE + dy
                        for dx in range(OLED_SCALE):
                            x = (col + i) * OLED_CELL + c * OLED_SCALE + dx
                            fb[(y // 8) * oled_view.OLED_W + x] |= 1 << (y % 8)
    return fb


def test_oled_boot_frame_is_the_ready_banner(oled_run, font_glyphs):
    """上电第一次出现内容时，画的必须是屏幕正中的英文 ready 横幅（自证固件跑起来了）。

    预期行为：横幅是两行（一行只有 OLED_COLS 格，"nanomeow ready" 放不下），垂直居中于正文区；
              这时其余页还是空的（用户输入要等收到一行才画）。
    """
    lines = ["nanomeow", "ready"]
    _, bus = oled_run
    assert bus.transactions > 0, "一条 I2C 事务都没有，GPIOB 钩子没接上"
    frames = [f for f in bus.frames if any(f)]
    assert frames, "整轮下来屏幕一直是黑的"

    # oled_view 记的是每个数据事务后的快照；一次 flush 会按脏页连发多个事务，所以
    # 「横幅刷完」的完整状态是用户输入开始画之前的那一帧，不是第一帧。
    banner = frames[0]
    for k, f in enumerate(frames):
        if k and any(f[:oled_view.OLED_W]):
            banner = frames[k - 1]
            break

    text_pages = oled_view.OLED_PAGES - OLED_LINE_PAGES
    top = (text_pages - len(lines) * OLED_LINE_PAGES) // 2
    covered = set()
    for i, line in enumerate(lines):
        page = top + i * OLED_LINE_PAGES
        covered.update(range(page, page + OLED_LINE_PAGES))
        got = bytes(banner[page * oled_view.OLED_W:(page + OLED_LINE_PAGES) * oled_view.OLED_W])
        left = (OLED_COLS - len(line)) // 2
        assert got == bytes(render_line(line, font_glyphs, col=left)), \
            "上电第一帧第 %d 行不是居中的 %r" % (i, line)
    for p in range(oled_view.OLED_PAGES):
        if p in covered:
            continue
        assert not any(banner[p * oled_view.OLED_W:(p + 1) * oled_view.OLED_W]), \
            "上电第一帧除了横幅还有别的内容（第 %d 页）" % p


def test_oled_keeps_the_user_input_on_screen(oled_run, font_glyphs):
    """用户输入必须完整画到正文区第一行上屏，不能被状态行或回复一上来就吃掉。

    预期行为：2x 放大后正文区只有 OLED_TEXT_PAGES 页（3 行），长回复会把整段正文正常上滚，
              所以这里查「用户输入完整上屏过」，而不是「最后一屏还在」。
    """
    _, bus = oled_run
    span = OLED_LINE_PAGES * oled_view.OLED_W
    assert any(bytes(f[:span]) == bytes(render_line(LINE, font_glyphs)) for f in bus.frames), \
        "整轮下来正文区第一行从没完整出现用户输入 %r" % LINE


def test_oled_status_row_puts_tps_at_the_right(oled_run):
    """状态行的 tps 必须贴右：最右边那一格有字（tps 长时整行都可能被占满）。"""
    _, bus = oled_run
    w = oled_view.OLED_W
    row = oled_view.OLED_PAGES - OLED_LINE_PAGES          # tps 画在状态行首行
    status = bytes(bus.frames[-1][row * w:(row + 1) * w])
    assert any(status[w - OLED_CELL:]), "状态行最右边一格没有画 tps"


def test_oled_keeps_redrawing_while_generating(oled_run):
    """生成是边吐边画：屏幕内容变化的次数要明显多于「上电 + 输入」这两步。"""
    _, bus = oled_run
    assert len(bus.frames) > 8, "屏幕只变了 %d 次，生成的字没画上去" % len(bus.frames)
    assert oled_view.count_lit(max(bus.frames, key=oled_view.count_lit)) > 0
