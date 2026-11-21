"""Annex B H.264 NAL-unit parsing utilities."""

from .bitstream import (
    MAX_NAL_BYTES,
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

__all__ = [
    "MAX_NAL_BYTES",
    "NaluError",
    "insert_emulation",
    "nal_type",
    "parse_batch",
    "parse_pps",
    "parse_sps",
    "split_nals",
    "strip_emulation",
    "validate_stream",
]
