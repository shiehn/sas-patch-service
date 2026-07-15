"""Surge XT .fxp read/write.

A Surge patch file is a VST2-style FXP "opaque chunk" program:
big-endian header (fxChunkSetCustom) followed by Surge's native patch stream
(the same 'sub3' stream Surge's plugin state contains). Verified against
SurgeSynthesizerIO.cpp at release_xt_1.3.4.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

CHUNK_MAGIC = b"CcnK"
FX_MAGIC = b"FPCh"  # opaque-chunk program
FX_ID = b"cjs3"

_HEADER = struct.Struct(">4si4si4sii28si")  # 60 bytes total
HEADER_SIZE = _HEADER.size


@dataclass
class Fxp:
    chunk: bytes  # Surge native patch stream ('sub3' ...)
    name: str = ""
    fx_version: int = 1

    def to_bytes(self) -> bytes:
        name_b = self.name.encode("utf-8", "replace")[:28].ljust(28, b"\x00")
        byte_size = HEADER_SIZE - 8 + len(self.chunk)  # everything after byteSize field
        header = _HEADER.pack(
            CHUNK_MAGIC, byte_size, FX_MAGIC, 1, FX_ID,
            self.fx_version, 1, name_b, len(self.chunk),
        )
        return header + self.chunk


def read(data: bytes) -> Fxp:
    if len(data) < HEADER_SIZE:
        raise ValueError("file too small for an fxp header")
    (chunk_magic, _byte_size, fx_magic, _version, fx_id,
     fx_version, _num_programs, name_b, chunk_size) = _HEADER.unpack_from(data, 0)
    if chunk_magic != CHUNK_MAGIC:
        raise ValueError(f"bad chunkMagic {chunk_magic!r}")
    if fx_magic != FX_MAGIC:
        raise ValueError(f"not an opaque-chunk fxp (fxMagic {fx_magic!r})")
    if fx_id != FX_ID:
        raise ValueError(f"not a Surge patch (fxID {fx_id!r})")
    chunk = data[HEADER_SIZE:HEADER_SIZE + chunk_size]
    if len(chunk) != chunk_size:
        raise ValueError(f"truncated chunk: header claims {chunk_size}, got {len(chunk)}")
    return Fxp(chunk=chunk, name=name_b.rstrip(b"\x00").decode("utf-8", "replace"),
               fx_version=fx_version)


def read_file(path) -> Fxp:
    with open(path, "rb") as f:
        return read(f.read())


def write_file(path, fxp: Fxp) -> None:
    with open(path, "wb") as f:
        f.write(fxp.to_bytes())
