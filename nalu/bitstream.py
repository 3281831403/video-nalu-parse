"""Annex B H.264 NAL-unit parsing primitives.

Byte-stream contract
--------------------
* A stream is an Annex B byte stream.  Each NAL unit is introduced by either
  the three-byte start code ``00 00 01`` or the four-byte start code
  ``00 00 00 01``.  Start codes are delimiters and never belong to the NAL
  payload.
* A NAL payload starts with its one-byte NAL header.  ``nal_unit_type`` is the
  low five bits of that byte (``header & 0x1F``); type 0 is forbidden.
* RBSP bytes are escaped with the H.264 emulation-prevention scheme: an
  ``0x03`` is inserted after ``00 00`` when the following byte is in
  ``00..03``.  Removing those ``0x03`` bytes must restore the RBSP exactly.
* All inputs and outputs are ``bytes`` objects; no file, network, random
  source, or mutable global buffer is used.
* Byte-sequence assertions allow a trailing tolerance of **1 byte** for
  Annex B ``trailing_zero_8bits``.  That tolerance never applies to the NAL
  header, a start code, or an inserted emulation-prevention byte: those must
  match exactly.  Bit fields inside an SPS/PPS are parsed MSB first.

The public functions are intentionally declared here; their implementation is
the bugfix target of ``tests/test_bitstream.py``.
"""

from __future__ import annotations


class NaluError(ValueError):
    """Raised when a NAL unit, SPS, PPS, or batch is malformed."""


# A single NAL larger than this is treated as a corrupt length by
# validate_stream.  The value is part of the public contract.
MAX_NAL_BYTES = 4 * 1024 * 1024

# profile_idc values whose SPS carries chroma_format_idc and the
# high-profile-only fields (scaling lists, bit depths).
_HIGH_PROFILES = frozenset(
    {100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135}
)

# chroma_format_idc -> (SubWidthC, SubHeightC)
_SUB_WIDTH_HEIGHT = {
    0: (1, 1),  # monochrome (ChromaArrayType == 0)
    1: (2, 2),  # 4:2:0
    2: (2, 1),  # 4:2:2
    3: (1, 1),  # 4:4:4
}


class _BitReader:
    """MSB-first bit reader over an RBSP, raising NaluError on truncation."""

    __slots__ = ("_data", "_bit")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._bit = 0

    def _read_bit(self) -> int:
        index = self._bit >> 3
        if index >= len(self._data):
            raise NaluError("truncated bitstream")
        value = (self._data[index] >> (7 - (self._bit & 7))) & 1
        self._bit += 1
        return value

    def read_bits(self, count: int) -> int:
        if count < 0:
            raise NaluError("negative bit count")
        value = 0
        for _ in range(count):
            value = (value << 1) | self._read_bit()
        return value

    def read_flag(self) -> bool:
        return bool(self._read_bit())

    def read_ue(self) -> int:
        zeros = 0
        while self._read_bit() == 0:
            zeros += 1
            if zeros > 31:
                raise NaluError("invalid Exp-Golomb code")
        if zeros == 0:
            return 0
        return (1 << zeros) - 1 + self.read_bits(zeros)

    def read_se(self) -> int:
        code_num = self.read_ue()
        if code_num & 1:
            return (code_num + 1) >> 1
        return -(code_num >> 1)


def _skip_scaling_list(reader: _BitReader, size: int) -> None:
    last_scale = 8
    next_scale = 8
    for _ in range(size):
        if next_scale != 0:
            next_scale = (last_scale + reader.read_se() + 256) % 256
        if next_scale != 0:
            last_scale = next_scale


def _split_rbsp(nal: bytes) -> bytes:
    """Return the de-escaped RBSP for a NAL payload (header byte included)."""
    return strip_emulation(nal)


def _payload_type(nal: bytes) -> int:
    """Raw ``header & 0x1F`` without the forbidden-type check."""
    return nal[0] & 0x1F


def _require_type(nal: bytes, expected: int, kind: str) -> None:
    if not nal:
        raise NaluError(f"empty {kind} NAL")
    actual = nal_type(nal)
    if actual != expected:
        raise NaluError(f"{kind} NAL has nal_unit_type {actual}, expected {expected}")


def split_nals(data: bytes) -> list[bytes]:
    """Split an Annex B byte stream into NAL payloads.

    Both the three-byte start code ``00 00 01`` and the four-byte start code
    ``00 00 00 01`` are recognised.  The delimiter is removed completely and
    the first returned byte of every payload is the NAL header byte (for
    example ``0x67`` for an SPS); the header must not be consumed as part of
    the start code.  A four-byte start code is not returned with a leading
    zero in the previous NAL.

    Empty input returns ``[]``.  Non-empty input that does not begin with a
    start code raises :class:`NaluError`.  A trailing start code produces a
    trailing empty payload so that :func:`validate_stream` can report the
    empty NAL instead of silently hiding it.  Annex B
    ``trailing_zero_8bits`` immediately before a following start code are
    delimiter padding and are not part of either NAL; byte-sequence
    comparisons allow a one-byte difference for that padding.
    """

    if not data:
        return []

    # Locate every start code: 00 00 01, with any extra leading zero bytes
    # (four-byte start code and trailing_zero_8bits) treated as delimiter.
    starts: list[tuple[int, int]] = []
    i = 0
    n = len(data)
    while i + 2 < n:
        if data[i] == 0 and data[i + 1] == 0 and data[i + 2] == 1:
            prefix = i
            while prefix > 0 and data[prefix - 1] == 0:
                prefix -= 1
            starts.append((prefix, i))
            i += 3
        else:
            i += 1

    if not starts:
        raise NaluError("data does not start with an Annex B start code")
    if starts[0][0] != 0:
        raise NaluError("data does not start with an Annex B start code")

    nals: list[bytes] = []
    for index, (_, payload_start) in enumerate(starts):
        if index + 1 < len(starts):
            end = starts[index + 1][0]
        else:
            end = n
        nals.append(data[payload_start + 3 : end])
    return nals


def nal_type(nal: bytes) -> int:
    """Return ``nal_unit_type`` from a NAL payload.

    The value is the low five bits of the NAL header, ``nal[0] & 0x1F`` -- not
    the high bits of the byte.  For example header ``0x65`` is type 5 (IDR
    slice) and header ``0x41`` is type 1 (non-IDR slice).

    An empty payload raises :class:`NaluError`.  ``nal_unit_type == 0`` is
    forbidden by H.264 and also raises :class:`NaluError`; callers must not
    continue parsing a type-0 NAL as though it had a defined layout.
    """

    if not nal:
        raise NaluError("empty NAL payload")
    value = nal[0] & 0x1F
    if value == 0:
        raise NaluError("forbidden nal_unit_type 0")
    return value


def strip_emulation(nal: bytes) -> bytes:
    """Remove emulation-prevention bytes from a NAL payload.

    Scanning left to right, when the current three encoded bytes are
    ``00 00 03``, write ``00 00`` and advance past the three bytes, then
    resume scanning at the following byte.  This removes exactly one
    ``emulation_prevention_three_byte`` and handles runs that overlap the
    following byte correctly.  In particular ``00 00 03 00`` becomes
    ``00 00 00`` and ``00 00 03 03`` becomes ``00 00 03``.  No byte other than
    an escape ``0x03`` is discarded.

    For any RBSP accepted by :func:`insert_emulation`, this function is its
    exact inverse.  Empty input returns ``b""``.
    """

    out = bytearray()
    i = 0
    n = len(nal)
    while i < n:
        if i + 2 < n and nal[i] == 0 and nal[i + 1] == 0 and nal[i + 2] == 3:
            out += b"\x00\x00"
            i += 3
        else:
            out.append(nal[i])
            i += 1
    return bytes(out)


def insert_emulation(rbsp: bytes) -> bytes:
    """Insert H.264 emulation-prevention bytes into an RBSP.

    A single ``0x03`` is inserted after ``00 00`` whenever the next RBSP byte
    is in ``00..03``.  A run of four zero bytes therefore becomes
    ``00 00 03 00 00`` -- exactly one escape at that emulation point, never
    two or more.  The result must contain no ``00 00 00``, ``00 00 01``, or
    ``00 00 02`` sequence; a ``00 00 03`` may appear only where the ``03`` is
    the inserted escape byte.  The result must satisfy
    ``strip_emulation(insert_emulation(rbsp)) == rbsp``.

    The input is not mutated; empty input returns ``b""``.
    """

    out = bytearray()
    zeros = 0
    for byte in rbsp:
        if zeros >= 2 and byte <= 3:
            out.append(3)
            zeros = 0
        out.append(byte)
        if byte == 0:
            zeros += 1
        else:
            zeros = 0
    return bytes(out)


def parse_sps(nal: bytes) -> dict:
    """Parse a complete SPS NAL payload (header byte plus RBSP) and return::

        {
            "sps_id": int,
            "profile_idc": int,
            "level_idc": int,
            "chroma_format_idc": int,
            "frame_mbs_only_flag": bool,
            "width": int,
            "height": int,
            "frame_cropping_flag": bool,
            "crop_left": int, "crop_right": int,
            "crop_top": int, "crop_bottom": int,
        }

    ``nal[0]`` must have ``nal_unit_type == 7``.  Exp-Golomb fields are decoded
    as ``ue(v)``; a truncated field, an invalid leading-zero run, a wrong NAL
    type, or a forbidden type raises :class:`NaluError`.  Emulation-prevention
    bytes are removed before bit parsing.

    The coded size is ``(pic_width_in_mbs_minus1 + 1) * 16`` by
    ``(2 - frame_mbs_only_flag) * (pic_height_in_map_units_minus1 + 1) * 16``.
    When ``frame_cropping_flag`` is set, the returned ``width`` and ``height``
    are the display size after subtracting the crop window, using
    ``CropUnitX = SubWidthC`` and
    ``CropUnitY = SubHeightC * (2 - frame_mbs_only_flag)`` for the SPS chroma
    format.  Ignoring this correction leaves black borders in the decoded
    picture.  Widths and heights are exact integer pixels; no byte/bit
    tolerance applies to them.
    """

    _require_type(nal, 7, "SPS")
    reader = _BitReader(_split_rbsp(nal))
    reader.read_bits(8)  # NAL header byte

    profile_idc = reader.read_bits(8)
    reader.read_bits(8)  # constraint_set flags + reserved bits
    level_idc = reader.read_bits(8)
    sps_id = reader.read_ue()

    chroma_format_idc = 1
    if profile_idc in _HIGH_PROFILES:
        chroma_format_idc = reader.read_ue()
        if chroma_format_idc > 3:
            raise NaluError("invalid chroma_format_idc")
        if chroma_format_idc == 3:
            reader.read_flag()  # separate_colour_plane_flag
        reader.read_ue()  # bit_depth_luma_minus8
        reader.read_ue()  # bit_depth_chroma_minus8
        reader.read_flag()  # qpprime_y_zero_transform_bypass_flag
        if reader.read_flag():  # seq_scaling_matrix_present_flag
            count = 12 if chroma_format_idc == 3 else 8
            for i in range(count):
                if reader.read_flag():  # seq_scaling_list_present_flag[i]
                    _skip_scaling_list(reader, 16 if i < 6 else 64)

    reader.read_ue()  # log2_max_frame_num_minus4
    pic_order_cnt_type = reader.read_ue()
    if pic_order_cnt_type == 0:
        reader.read_ue()  # log2_max_pic_order_cnt_lsb_minus4
    elif pic_order_cnt_type == 1:
        reader.read_flag()  # delta_pic_order_always_zero_flag
        reader.read_se()  # offset_for_non_ref_pic
        reader.read_se()  # offset_for_top_to_bottom_field
        cycle = reader.read_ue()
        for _ in range(cycle):
            reader.read_se()  # offset_for_ref_frame[i]
    elif pic_order_cnt_type != 2:
        raise NaluError("invalid pic_order_cnt_type")

    reader.read_ue()  # max_num_ref_frames
    reader.read_flag()  # gaps_in_frame_num_value_allowed_flag

    pic_width_in_mbs_minus1 = reader.read_ue()
    pic_height_in_map_units_minus1 = reader.read_ue()
    frame_mbs_only_flag = reader.read_flag()
    if not frame_mbs_only_flag:
        reader.read_flag()  # mb_adaptive_frame_field_flag
    reader.read_flag()  # direct_8x8_inference_flag

    frame_cropping_flag = reader.read_flag()
    crop_left = crop_right = crop_top = crop_bottom = 0
    if frame_cropping_flag:
        crop_left = reader.read_ue()
        crop_right = reader.read_ue()
        crop_top = reader.read_ue()
        crop_bottom = reader.read_ue()

    width = (pic_width_in_mbs_minus1 + 1) * 16
    height = (2 - int(frame_mbs_only_flag)) * (
        pic_height_in_map_units_minus1 + 1
    ) * 16

    if chroma_format_idc == 0:
        crop_unit_x = 1
        crop_unit_y = 2 - int(frame_mbs_only_flag)
    else:
        sub_width_c, sub_height_c = _SUB_WIDTH_HEIGHT[chroma_format_idc]
        crop_unit_x = sub_width_c
        crop_unit_y = sub_height_c * (2 - int(frame_mbs_only_flag))

    width -= (crop_left + crop_right) * crop_unit_x
    height -= (crop_top + crop_bottom) * crop_unit_y
    if width <= 0 or height <= 0:
        raise NaluError("frame cropping removes the whole picture")

    return {
        "sps_id": sps_id,
        "profile_idc": profile_idc,
        "level_idc": level_idc,
        "chroma_format_idc": chroma_format_idc,
        "frame_mbs_only_flag": frame_mbs_only_flag,
        "width": width,
        "height": height,
        "frame_cropping_flag": frame_cropping_flag,
        "crop_left": crop_left,
        "crop_right": crop_right,
        "crop_top": crop_top,
        "crop_bottom": crop_bottom,
    }


def parse_pps(nal: bytes) -> dict:
    """Parse a complete PPS NAL payload (header byte plus RBSP) and return::

        {
            "pps_id": int,
            "sps_id": int,
            "entropy_coding_mode_flag": bool,
            "bottom_field_pic_order_in_frame_present_flag": bool,
            "num_slice_groups_minus1": int,
            "num_ref_idx_l0_default_active_minus1": int,
            "num_ref_idx_l1_default_active_minus1": int,
            "weighted_pred_flag": bool,
            "weighted_bipred_idc": int,
        }

    ``nal[0]`` must have ``nal_unit_type == 8``.  ``pic_parameter_set_id`` and
    ``seq_parameter_set_id`` are Exp-Golomb ``ue(v)`` fields, not fixed-width
    integers, so leading zero bits are significant.  ``pps_id`` must be in
    ``0..255`` and ``sps_id`` in ``0..31``.  A wrong NAL type, truncation, an
    out-of-range id, or an invalid Exp-Golomb code raises :class:`NaluError`.
    ``entropy_coding_mode_flag`` is returned as a ``bool`` (``True`` means
    CABAC).  Emulation-prevention bytes are removed before bit parsing.
    """

    _require_type(nal, 8, "PPS")
    reader = _BitReader(_split_rbsp(nal))
    reader.read_bits(8)  # NAL header byte

    pps_id = reader.read_ue()
    sps_id = reader.read_ue()
    if pps_id > 255:
        raise NaluError("pic_parameter_set_id out of range")
    if sps_id > 31:
        raise NaluError("seq_parameter_set_id out of range")

    entropy_coding_mode_flag = reader.read_flag()
    bottom_field_pic_order_in_frame_present_flag = reader.read_flag()
    num_slice_groups_minus1 = reader.read_ue()
    if num_slice_groups_minus1 > 7:
        raise NaluError("num_slice_groups_minus1 out of range")
    if num_slice_groups_minus1 > 0:
        slice_group_map_type = reader.read_ue()
        if slice_group_map_type > 6:
            raise NaluError("invalid slice_group_map_type")
        if slice_group_map_type == 0:
            for _ in range(num_slice_groups_minus1 + 1):
                reader.read_ue()  # run_length_minus1[i]
        elif slice_group_map_type == 2:
            for _ in range(num_slice_groups_minus1):
                reader.read_ue()  # top_left[i]
                reader.read_ue()  # bottom_right[i]
        elif slice_group_map_type in (3, 4, 5):
            reader.read_flag()  # slice_group_change_direction_flag
            reader.read_ue()  # slice_group_change_rate_minus1
        elif slice_group_map_type == 6:
            pic_size_in_map_units_minus1 = reader.read_ue()
            bits = max(1, (num_slice_groups_minus1).bit_length())
            for _ in range(pic_size_in_map_units_minus1 + 1):
                reader.read_bits(bits)  # slice_group_id[i]

    num_ref_idx_l0_default_active_minus1 = reader.read_ue()
    num_ref_idx_l1_default_active_minus1 = reader.read_ue()
    weighted_pred_flag = reader.read_flag()
    weighted_bipred_idc = reader.read_bits(2)
    reader.read_se()  # pic_init_qp_minus26
    reader.read_se()  # pic_init_qs_minus26
    reader.read_se()  # chroma_qp_index_offset
    reader.read_flag()  # deblocking_filter_control_present_flag
    reader.read_flag()  # constrained_intra_pred_flag
    reader.read_flag()  # redundant_pic_cnt_present_flag

    return {
        "pps_id": pps_id,
        "sps_id": sps_id,
        "entropy_coding_mode_flag": entropy_coding_mode_flag,
        "bottom_field_pic_order_in_frame_present_flag": (
            bottom_field_pic_order_in_frame_present_flag
        ),
        "num_slice_groups_minus1": num_slice_groups_minus1,
        "num_ref_idx_l0_default_active_minus1": (
            num_ref_idx_l0_default_active_minus1
        ),
        "num_ref_idx_l1_default_active_minus1": (
            num_ref_idx_l1_default_active_minus1
        ),
        "weighted_pred_flag": weighted_pred_flag,
        "weighted_bipred_idc": weighted_bipred_idc,
    }


def validate_stream(data: bytes) -> list[str]:
    """Return structural problems found in an Annex B byte stream.

    An empty byte stream is valid and returns ``[]``.  A non-empty stream must
    contain at least one start code.  For each NAL payload, in stream order,
    the returned list contains zero or more of these exact issue strings:

    * ``"missing start code"`` -- non-empty input has no Annex B delimiter.
    * ``"empty NAL at index {i}"`` -- two adjacent start codes, or a trailing
      start code, produced a zero-length payload.
    * ``"forbidden nal_unit_type 0 at index {i}"`` -- type 0 is not defined.
    * ``"NAL length out of bounds at index {i}"`` -- the payload has fewer
      than 2 bytes (header plus at least one RBSP byte) or is larger than
      :data:`MAX_NAL_BYTES`.

    Validation is a reporting operation: malformed NALs are listed rather
    than raised.  The returned strings must be deterministic and ordered by
    NAL index.
    """

    if not data:
        return []
    try:
        nals = split_nals(data)
    except NaluError:
        return ["missing start code"]

    issues: list[str] = []
    for index, nal in enumerate(nals):
        if not nal:
            issues.append(f"empty NAL at index {index}")
            continue
        payload_type = _payload_type(nal)
        if payload_type == 0:
            issues.append(f"forbidden nal_unit_type 0 at index {index}")
        if len(nal) < 2 or len(nal) > MAX_NAL_BYTES:
            issues.append(f"NAL length out of bounds at index {index}")
    return issues


def parse_batch(streams: list[tuple[str, bytes]]) -> dict:
    """Parse a batch of named streams atomically and return::

        {
            "order": [name, ...],                 # input order
            "streams": {
                name: {
                    "nal_types": [int, ...],
                    "sps": dict,                  # parse_sps result
                    "pps": dict,                  # parse_pps result
                },
                ...
            },
        }

    ``streams`` is an ordered sequence of ``(name, data)`` pairs with unique
    names.  Every stream is validated with :func:`validate_stream`, then must
    contain exactly one SPS and exactly one PPS; the returned ``nal_types``
    lists all NAL types in stream order.  Each stream is parsed into its own
    fresh objects -- no cursor, byte buffer, SPS dict, or PPS dict may be
    shared between entries, otherwise a later stream can overwrite an earlier
    result.

    If any name is duplicated, any stream is invalid, or any required SPS/PPS
    is missing or ambiguous, the whole batch raises :class:`NaluError` and no
    partial result is returned or retained.  An empty batch returns
    ``{"order": [], "streams": {}}``.  Successful results preserve input
    order in both ``order`` and the insertion order of ``streams``.
    """

    if not streams:
        return {"order": [], "streams": {}}

    seen: set[str] = set()
    for name, _ in streams:
        if name in seen:
            raise NaluError(f"duplicate stream name: {name!r}")
        seen.add(name)

    staged: dict[str, dict] = {}
    for name, data in streams:
        issues = validate_stream(data)
        if issues:
            raise NaluError(f"invalid stream {name!r}: {'; '.join(issues)}")

        nals = split_nals(data)
        nal_types: list[int] = []
        sps_nals: list[bytes] = []
        pps_nals: list[bytes] = []
        for nal in nals:
            value = nal_type(nal)
            nal_types.append(value)
            if value == 7:
                sps_nals.append(nal)
            elif value == 8:
                pps_nals.append(nal)

        if len(sps_nals) != 1:
            raise NaluError(
                f"stream {name!r} has {len(sps_nals)} SPS NALs, expected 1"
            )
        if len(pps_nals) != 1:
            raise NaluError(
                f"stream {name!r} has {len(pps_nals)} PPS NALs, expected 1"
            )

        # Fresh objects per stream: nothing is shared between entries.
        staged[name] = {
            "nal_types": nal_types,
            "sps": parse_sps(sps_nals[0]),
            "pps": parse_pps(pps_nals[0]),
        }

    return {"order": [name for name, _ in streams], "streams": staged}
