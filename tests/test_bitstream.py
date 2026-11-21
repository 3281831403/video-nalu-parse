"""video-nalu-parse 的验收测试（不可修改）。

所有 NAL 与码流都用 ``bytes`` 手工构造，不访问文件、网络，也不使用随机数。
修复前，15 条测试全部因 ``nalu.bitstream`` 的桩函数抛出
``NotImplementedError`` 而失败；测试本身不依赖被测代码来生成输入。
"""

import pytest

from nalu.bitstream import (
    NaluError,
    insert_emulation,
    nal_type,
    parse_batch,
    parse_pps,
    parse_sps,
    split_nals,
    strip_emulation,
    validate_stream,
)


# ---------------------------------------------------------------------------
# 手工构造的码流片段（静态 bytes，不调用被测函数）
# ---------------------------------------------------------------------------

# baseline profile 66, level 3.0, 640x480 coded size, frame_crop_bottom_offset=8.
SPS_640X480_CROP_BOTTOM_8 = bytes.fromhex("6742001ef40501efc4a0")
# baseline profile 66, level 2.0, 320x240, no cropping.
SPS_320X240 = bytes.fromhex("67420014f40a0fc8")
# pic_parameter_set_id=3, seq_parameter_set_id=0, CABAC enabled.
PPS_ID_3_CABAC = bytes.fromhex("6826e3c8")
# pic_parameter_set_id=4, seq_parameter_set_id=0, CAVLC.
PPS_ID_4_CAVLC = bytes.fromhex("682ce3c8")


def _annex_b(*nals: bytes) -> bytes:
    """用 4 字节起始码拼一条 Annex B 码流。"""
    return b"".join(b"\x00\x00\x00\x01" + nal for nal in nals)


# ---------------------------------------------------------------------------
# 10 个缺陷各一条
# ---------------------------------------------------------------------------


def test_split_nals_accepts_three_byte_start_code():
    """缺陷 1：只认 4 字节起始码，3 字节起始码的 NAL 漏切。"""
    data = (
        b"\x00\x00\x00\x01\x67\x42\x00\x1e"
        b"\x00\x00\x01\x68\xce\x3c\x80"
    )

    assert split_nals(data) == [
        b"\x67\x42\x00\x1e",
        b"\x68\xce\x3c\x80",
    ]


def test_split_nals_preserves_nal_header():
    """缺陷 2：去掉起始码时多切 1 字节，NAL 头丢失。"""
    nals = split_nals(b"\x00\x00\x01\x65\x88\x84")

    assert nals == [b"\x65\x88\x84"]
    assert nals[0][0] == 0x65


def test_nal_type_uses_low_five_bits():
    """缺陷 3：类型取高 5 位，所有 nal_unit_type 都错了。"""
    assert nal_type(b"\x65\x88\x84") == 5
    assert nal_type(b"\x67\x42\x00\x1e") == 7
    assert nal_type(b"\x68\xce\x3c\x80") == 8


def test_strip_emulation_restores_data_after_03():
    """缺陷 4：只删 03 但没恢复后续数据，RBSP 被截断。"""
    escaped = b"\x65\x00\x00\x03\x00\x00\x00\x03\x01\x00\x00\x03\x03"

    assert strip_emulation(escaped) == (
        b"\x65\x00\x00\x00\x00\x00\x01\x00\x00\x03"
    )


def test_insert_emulation_escapes_zero_run_once():
    """缺陷 5：对 00 00 00 连续插 03，转义后的编码不合法。

    一个零串只允许在真正形成 00 00 00/01/02/03 的位置插入一个 03，
    不能在插过一次后继续对新增的 00 再插。
    """
    assert insert_emulation(b"\x00\x00\x00") == b"\x00\x00\x03\x00"
    escaped = insert_emulation(b"\x00\x00\x00\x01")

    assert escaped == b"\x00\x00\x03\x00\x01"
    assert escaped.count(b"\x03") == 1


def test_parse_sps_applies_frame_cropping():
    """缺陷 6：SPS 宽高不按裁剪修正，画面多出黑边。"""
    sps = parse_sps(SPS_640X480_CROP_BOTTOM_8)

    assert sps["profile_idc"] == 66
    assert sps["level_idc"] == 30
    assert sps["frame_cropping_flag"] is True
    assert (sps["crop_left"], sps["crop_right"]) == (0, 0)
    assert (sps["crop_top"], sps["crop_bottom"]) == (0, 8)
    assert sps["width"] == 640
    assert sps["height"] == 464


def test_parse_pps_reads_exp_golomb_ids():
    """缺陷 7：PPS id 用定长读取而不是 ue(v)，id 解析错误。"""
    pps = parse_pps(PPS_ID_3_CABAC)

    assert pps["pps_id"] == 3
    assert pps["sps_id"] == 0
    assert pps["entropy_coding_mode_flag"] is True


def test_validate_stream_reports_empty_nal():
    """缺陷 8：空 NAL 不报错，后续读取越界。"""
    data = b"\x00\x00\x01\x00\x00\x01\x67\x00"

    assert validate_stream(data) == ["empty NAL at index 0"]
    assert validate_stream(b"\x00\x00\x01\x67") == [
        "NAL length out of bounds at index 0"
    ]


def test_parse_batch_keeps_streams_independent():
    """缺陷 9：批量共用缓冲，后一条流覆盖前一条流。"""
    stream_a = _annex_b(SPS_640X480_CROP_BOTTOM_8, PPS_ID_3_CABAC)
    stream_b = _annex_b(SPS_320X240, PPS_ID_4_CAVLC)

    batch = parse_batch([("first", stream_a), ("second", stream_b)])

    first = batch["streams"]["first"]
    second = batch["streams"]["second"]
    assert first is not second
    assert first["sps"]["width"] == 640
    assert first["sps"]["height"] == 464
    assert first["pps"]["pps_id"] == 3
    assert second["sps"]["width"] == 320
    assert second["sps"]["height"] == 240
    assert second["pps"]["pps_id"] == 4


def test_nal_type_rejects_forbidden_zero():
    """缺陷 10：类型 0 不报错，进入未定义解析。"""
    with pytest.raises(NaluError):
        nal_type(b"\x00\x00")

    issues = validate_stream(b"\x00\x00\x01\x00\x00")
    assert issues == ["forbidden nal_unit_type 0 at index 0"]


# ---------------------------------------------------------------------------
# 另外 5 条
# ---------------------------------------------------------------------------


def test_split_nals_rejects_payload_without_start_code():
    """单 NAL 无起始码输入必须抛 NaluError，而不是当成裸 NAL。"""
    with pytest.raises(NaluError):
        split_nals(b"\x65\x88\x84")


def test_emulation_roundtrip_is_exact():
    """insert_emulation 后 strip_emulation 必须逐字节还原。"""
    rbsp = b"\x65\x00\x00\x00\x00\x00\x01\x00\x00\x02\x00\x00\x03\x80"

    escaped = insert_emulation(rbsp)

    assert strip_emulation(escaped) == rbsp
    assert b"\x00\x00\x01" not in escaped
    assert b"\x00\x00\x02" not in escaped


def test_empty_stream_and_batch_return_empty():
    """空流返回空，空批次也返回空结构。"""
    assert split_nals(b"") == []
    assert validate_stream(b"") == []
    assert parse_batch([]) == {"order": [], "streams": {}}


def test_type_one_is_a_slice():
    """类型 1 是非 IDR 切片，不能取成其他类型。"""
    nal = split_nals(b"\x00\x00\x01\x41\x9a")[0]

    assert nal == b"\x41\x9a"
    assert nal_type(nal) == 1


def test_parse_batch_preserves_order_and_rolls_back():
    """批量全成功时顺序不变；任一流非法时整体回滚。"""
    stream_a = _annex_b(SPS_640X480_CROP_BOTTOM_8, PPS_ID_3_CABAC)
    stream_b = _annex_b(SPS_320X240, PPS_ID_4_CAVLC)

    batch = parse_batch([("z", stream_b), ("a", stream_a)])

    assert batch["order"] == ["z", "a"]
    assert list(batch["streams"]) == ["z", "a"]
    assert batch["streams"]["z"]["nal_types"] == [7, 8]
    assert batch["streams"]["a"]["nal_types"] == [7, 8]

    with pytest.raises(NaluError):
        parse_batch([("ok", stream_a), ("bad", _annex_b(b"\x00\x00"))])
