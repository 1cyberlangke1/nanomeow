# -*- coding: utf-8 -*-
"""OLED 可视化模拟的纯函数闸门：I2C 位流解码、SSD1306 命令解析、字符画渲染。

输入：无（纯 Python，不碰 Unicorn、不碰固件）。
输出：pytest 断言。
预期行为：这三块是「屏上显示对不对」的全部逻辑，先在纯函数层面钉死；真固件跑出来的整条链
          （GPIOB 位流 → 解码 → 显示 RAM）在 test_m3_emulation.py 里对拍。
"""

import pathlib
import sys

import pytest

FW_DIR = pathlib.Path(__file__).resolve().parents[1] / "firmware"
sys.path.insert(0, str(FW_DIR))
import oled_view  # noqa: E402


class PinDriver:
    """按 nm_oled_port.c 的时序驱动 SCL / SDA，喂给一个 I2cDecoder。

    输入：无。
    输出：start() / stop() / put(byte)，以及收到的字节列表 got。
    预期行为：位序与 ACK 时钟和固件完全一致 —— 固件怎么发，这里就怎么发，才能证明解码器对得上。
    """

    def __init__(self):
        self.got = []
        self.dec = oled_view.I2cDecoder(self.got.append)
        self.scl = 0
        self.sda = 0

    def _set(self, scl=None, sda=None):
        """输入：要改的引脚电平；输出：无。预期行为：只把变化喂给解码器。"""
        if scl is not None:
            self.scl = scl
        if sda is not None:
            self.sda = sda
        self.dec.update(self.scl, self.sda)

    def start(self):
        """输入：无；输出：无。预期行为：与 nm_i2c_start 同序 —— SCL 高时 SDA 由高变低。"""
        self._set(sda=1)
        self._set(scl=1)
        self._set(sda=0)
        self._set(scl=0)

    def stop(self):
        """输入：无；输出：无。预期行为：与 nm_i2c_stop 同序 —— SCL 高时 SDA 由低变高。"""
        self._set(sda=0)
        self._set(scl=1)
        self._set(sda=1)

    def put(self, byte):
        """输入：一个字节；输出：第 9 个时钟上从机有没有应答（本模型永远应答）。"""
        for i in range(7, -1, -1):
            self._set(sda=(byte >> i) & 1)
            self._set(scl=1)
            self._set(scl=0)
        self._set(sda=1)                 # 释放 SDA，等从机拉低
        self._set(scl=1)
        acked = self.dec.ack_phase
        self._set(scl=0)
        return acked


def test_decoder_recovers_a_transaction():
    """起始 → 地址 → 控制字节 → 载荷 → 停止，字节要一个不少地还原出来。"""
    d = PinDriver()
    d.start()
    for b in (0x78, 0x00, 0xAE, 0xD5, 0x80):
        assert d.put(b), "第 9 个时钟没有进 ACK 相位"
    d.stop()
    assert d.got == [0x78, 0x00, 0xAE, 0xD5, 0x80]


def test_decoder_handles_back_to_back_transactions():
    """两次事务连着发（固件每一帧都这么干），第二次不能被第一次的尾巴带歪。"""
    d = PinDriver()
    d.start()
    for b in (0x78, 0x40, 0xFF):
        d.put(b)
    d.stop()
    d.start()
    for b in (0x78, 0x00, 0xAF):
        d.put(b)
    d.stop()
    assert d.got == [0x78, 0x40, 0xFF, 0x78, 0x00, 0xAF]


def test_decoder_ignores_bits_outside_a_transaction():
    """没有 START 之前的电平变化（上电把两个脚拉高）不能算数据位。"""
    d = PinDriver()
    d._set(sda=1)
    d._set(scl=1)
    d._set(scl=0)
    assert d.got == [], "START 之前就解出字节了"


def test_decoder_leaves_ack_phase_after_the_ninth_clock():
    """ACK 相位只在第 9 个时钟里成立；SCL 拉低后必须结束，否则固件会读到假的低电平。"""
    d = PinDriver()
    d.start()
    d.put(0x78)
    assert not d.dec.ack_phase


def test_screen_writes_a_page_in_horizontal_mode():
    """0x21 / 0x22 开一个单页窗口后，128 个数据字节要正好铺满那一页并绕回起点。"""
    scr = oled_view.OledScreen()
    for b in (0x20, 0x00, 0x21, 0x00, 0x7F, 0x22, 0x03, 0x03):
        scr.command(b)
    for i in range(128):
        scr.data(i)
    assert bytes(scr.ram[3 * 128:4 * 128]) == bytes(range(128))
    assert scr.ram[:3 * 128] == bytearray(3 * 128), "写一页却动了别的页"
    assert (scr.col, scr.page) == (0, 3), "128 个字节没有绕回窗口起点"


def test_screen_swallows_command_arguments():
    """命令的参数不能被当成命令：0xDA 的参数 0x12 是「高列起始」，被当命令执行光标就跑到第 32 列。"""
    scr = oled_view.OledScreen()
    for b in (0x20, 0x00, 0x21, 0x00, 0x7F, 0xDA, 0x12):
        scr.command(b)
    scr.data(0xA5)
    assert scr.ram[0] == 0xA5, "参数被当成命令吃掉了"


def test_render_screen_draws_pixels():
    """一个像素画一个字符：半块图里 (0,0) 是 '▀'，纯 ASCII 图里是 '#'。"""
    ram = bytearray(128 * 8)
    ram[0] = 0x01
    assert oled_view.count_lit(ram) == 1
    art = oled_view.render_screen(ram).split("\n")
    assert len(art) == 32 and len(art[0]) == 128
    assert art[0][0] == "▀"
    plain = oled_view.render_screen(ram, plain=True).split("\n")
    assert len(plain) == 64 and plain[0][0] == "#" and plain[1][0] == "."


def test_render_screen_rejects_short_ram():
    """显示 RAM 不够一屏时要报错，不能静默画出一张半截的图。"""
    with pytest.raises(ValueError):
        oled_view.render_screen(bytearray(10))
