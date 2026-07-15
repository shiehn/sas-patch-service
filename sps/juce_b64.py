"""JUCE MemoryBlock "dot-base64" codec.

Tracktion stores an external plugin's state in the <PLUGIN state="..."> attribute
using juce::MemoryBlock::toBase64Encoding(): "<decimal byte count>." followed by a
custom-alphabet base64 where each character carries 6 bits packed LSB-first into
the byte stream (see juce_MemoryBlock.cpp setBitRange/toBase64Encoding).

This is NOT RFC 4648 base64 — different alphabet, different bit order.
"""

from __future__ import annotations

# Alphabet from juce_MemoryBlock.cpp (getEncodingTable).
_TABLE = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"
_REVERSE = {c: i for i, c in enumerate(_TABLE)}


def decode(s: str) -> bytes:
    """Decode a JUCE MemoryBlock base64 string ("<size>.<payload>") to bytes."""
    size_str, sep, payload = s.partition(".")
    if not sep:
        raise ValueError("not a JUCE MemoryBlock base64 string (no '.' separator)")
    n = int(size_str)
    out = bytearray(n)
    pos = 0
    for ch in payload:
        v = _REVERSE.get(ch)
        if v is None:  # tolerate stray whitespace the way JUCE ignores unknown chars
            continue
        byte = pos >> 3
        bit = pos & 7
        if byte < n:
            out[byte] |= (v << bit) & 0xFF
            if bit > 2 and byte + 1 < n:
                out[byte + 1] |= v >> (8 - bit)
        pos += 6
    return bytes(out)


def encode(data: bytes) -> str:
    """Encode bytes to the JUCE MemoryBlock base64 string format."""
    n = len(data)
    num_chars = ((n << 3) + 5) // 6
    chars = []
    for i in range(num_chars):
        byte = (i * 6) >> 3
        start_bit = (i * 6) & 7
        bits = data[byte] >> start_bit
        if start_bit > 2 and byte + 1 < n:
            bits |= data[byte + 1] << (8 - start_bit)
        chars.append(_TABLE[bits & 63])
    return f"{n}.{''.join(chars)}"
