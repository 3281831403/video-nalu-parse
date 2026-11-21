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

    raise NotImplementedError


def nal_type(nal: bytes) -> int:
    """Return ``nal_unit_type`` from a NAL payload.

    The value is the low five bits of the NAL header, ``nal[0] & 0x1F`` -- not
    the high bits of the byte.  For example header ``0x65`` is type 5 (IDR
    slice) and header ``0x41`` is type 1 (non-IDR slice).

    An empty payload raises :class:`NaluError`.  ``nal_unit_type == 0`` is
    forbidden by H.264 and also raises :class:`NaluError`; callers must not
    continue parsing a type-0 NAL as though it had a defined layout.
    """

    raise NotImplementedError


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

    raise NotImplementedError


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

    raise NotImplementedError


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

    raise NotImplementedError


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

    raise NotImplementedError


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

    raise NotImplementedError


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

    raise NotImplementedError
