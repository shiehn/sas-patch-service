"""XML-level patch traits surgepy cannot reach — currently LFO tempo-sync.

Surge serializes per-parameter `temposync` as an XML attribute on the param
node (SurgePatch.cpp writes/reads it directly); surgepy 1.3.4 exposes no
binding for it, so the only route is editing the patch stream's XML. Node
naming (verified against release_xt_1.3.4 factory patches): `<a_lfo0_rate>`
… `<a_lfo11_rate>` — scene prefix 'a'/'b', ZERO-indexed LFO slots where
0–5 are the voice LFOs (UI "LFO 1–6") and 6–11 the scene S-LFOs.

Edits are byte-minimal (regex on the self-closed param node, everything else
verbatim) and the stream header's xmlsize field is recomputed; wavetable
blocks after the XML are preserved untouched.

CAMPAIGN USAGE: apply to SEEDS **before** an evolution campaign — fitness
renders hear the sync (a synced LFO 1 rate reads as a note division, e.g.
"1/8"), and every child inherits it via loadPatch(parent). Applying only to
survivors would ship a trait fitness never heard.
"""

from __future__ import annotations

import re
import struct
from typing import Iterable

from . import fxp as fxp_mod
from .wrapper import parse_patch_stream

_TEMPOSYNC_ATTR = re.compile(rb'\s*temposync="[^"]*"')


def _rate_node_pattern(scene: str, lfo_index0: int) -> re.Pattern[bytes]:
    return re.compile(
        rb"<" + f"{scene}_lfo{lfo_index0}_rate".encode("ascii") + rb"\b[^>]*/>"
    )


def set_lfo_temposync_xml(xml: bytes, lfo: int, enabled: bool = True, scene: str = "a") -> bytes:
    """Set/clear temposync on one LFO's rate param inside a patch XML document.

    `lfo` is the 1-based UI number of a VOICE LFO (1..6); `scene` is 'a'|'b'.
    Raises ValueError when the node is missing (malformed/foreign XML) so a
    silent no-op never masquerades as success.
    """
    if not 1 <= lfo <= 6:
        raise ValueError(f"lfo must be 1..6 (voice LFOs), got {lfo}")
    if scene not in ("a", "b"):
        raise ValueError(f"scene must be 'a' or 'b', got {scene!r}")

    pattern = _rate_node_pattern(scene, lfo - 1)
    match = pattern.search(xml)
    if match is None:
        raise ValueError(f"no <{scene}_lfo{lfo - 1}_rate> node in patch XML")

    node = _TEMPOSYNC_ATTR.sub(b"", match.group(0))
    if enabled:
        node = node[:-2].rstrip() + b' temposync="1" />'
    return xml[: match.start()] + node + xml[match.end():]


def set_lfo_temposync_stream(stream: bytes, lfos: Iterable[int], enabled: bool = True,
                             scene: str = "a") -> bytes:
    """Same edit on a full 'sub3' patch stream (fxp chunk / plugin state),
    recomputing the header's xmlsize and preserving wavetable blocks."""
    parsed = parse_patch_stream(stream)
    xml = parsed.xml
    for lfo in lfos:
        xml = set_lfo_temposync_xml(xml, lfo, enabled=enabled, scene=scene)

    raw = parsed.raw
    xml_start = raw.find(b"<?xml")
    if xml_start < 0:
        raise ValueError("no XML document found in patch stream")
    trailing = raw[xml_start + len(parsed.xml):]
    return raw[:4] + struct.pack("<i", len(xml)) + raw[8:xml_start] + xml + trailing


def apply_to_fxp_file(path_in: str, path_out: str, lfos: Iterable[int] = (1,),
                      enabled: bool = True, scene: str = "a") -> None:
    """Read an .fxp, set/clear temposync on the given voice LFOs, write it out."""
    f = fxp_mod.read_file(path_in)
    f.chunk = set_lfo_temposync_stream(f.chunk, lfos, enabled=enabled, scene=scene)
    fxp_mod.write_file(path_out, f)
