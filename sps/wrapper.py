"""Tracktion <PLUGIN> wrapper ⇄ Surge patch stream.

S&S bundled presets (sas-app/resources/tracktion-presets/SurgeXT/*.json) are maps of
{ "Preset_N": "<standard base64>" }. Empirically verified nesting (2026-07-15, against
real bundled presets and factory .fxp files):

  JSON value
   └─ std-base64 → <PLUGIN type="vst" uniqueId="190e4fbd" ... state="..."/>   (Tracktion)
       └─ state attr: JUCE MemoryBlock dot-base64 → bytes:
           'VC2!' + u32-LE xmlLen + XML text + trailing NUL      (AudioProcessor::
            copyXmlToBinary, magic 0x21324356)                    juce hosting layer)
            └─ <VST3PluginState><IComponent>dot-base64</IComponent></VST3PluginState>
                └─ IComponent bytes == Surge's native 'sub3' patch stream, VERBATIM —
                   the exact same bytes an .fxp file carries after its 60-byte header.

So: fxp chunk ⇄ client-ready preset state is purely mechanical re-wrapping. This module
implements every layer so the corpus pipeline can treat everything as .fxp and the
(future) service can emit exactly what the S&S client already applies.
"""

from __future__ import annotations

import base64
import re
import struct
from dataclasses import dataclass, field
from typing import Dict, Optional
from xml.etree import ElementTree as ET

from . import juce_b64

_XML_DECL = '<?xml version="1.0" encoding="UTF-8"?>'
_VC2_MAGIC = b"VC2!"  # juce AudioProcessor::copyXmlToBinary magic 0x21324356 (LE)

# Attributes observed on every bundled preset; used when synthesizing new wrappers.
DEFAULT_PLUGIN_ATTRS: Dict[str, str] = {
    "type": "vst",
    "uniqueId": "190e4fbd",
    "filename": "/Library/Audio/Plug-Ins/VST3/Surge XT.vst3",
    "name": "Surge XT",
    "manufacturer": "Surge Synth Team",
    "enabled": "1",
}


# ---- Layer 3: juce copyXmlToBinary container ('VC2!') -------------------------

def unwrap_vst3_state(state: bytes) -> bytes:
    """VC2! container → IComponent bytes (Surge's 'sub3' stream)."""
    if state[:4] != _VC2_MAGIC:
        raise ValueError(f"not a copyXmlToBinary block (magic {state[:4]!r})")
    (xml_len,) = struct.unpack_from("<I", state, 4)
    xml_text = state[8:8 + xml_len].decode("utf-8", "replace").rstrip("\x00")
    root = ET.fromstring(xml_text)
    if root.tag != "VST3PluginState":
        raise ValueError(f"expected <VST3PluginState>, got <{root.tag}>")
    comp = root.find("IComponent")
    if comp is None or not (comp.text or "").strip():
        raise ValueError("no <IComponent> payload")
    return juce_b64.decode(comp.text.strip())


def wrap_vst3_state(chunk: bytes) -> bytes:
    """Surge 'sub3' stream → VC2! container (what the hosting layer expects).

    Mirrors juce AudioProcessor::copyXmlToBinary exactly: magic + u32 length +
    single-line XML ("<?xml ...?> <element .../>") + trailing NUL, where the
    stored length EXCLUDES the NUL (JUCE writes totalSize - 9).
    """
    xml_text = (
        f"{_XML_DECL} "
        f"<VST3PluginState><IComponent>{juce_b64.encode(chunk)}</IComponent></VST3PluginState>"
    )
    xml_bytes = xml_text.encode("utf-8")
    return _VC2_MAGIC + struct.pack("<I", len(xml_bytes)) + xml_bytes + b"\x00"


# ---- Layer 2: Tracktion <PLUGIN> element --------------------------------------

@dataclass
class Wrapper:
    chunk: bytes                       # raw Surge patch stream ('sub3' ...)
    attrs: Dict[str, str] = field(default_factory=dict)  # <PLUGIN> attrs minus `state`
    raw_state: bytes = b""             # full VC2! container as found


def decode_wrapper_b64(preset_value: str) -> Wrapper:
    """Decode a bundled-preset JSON value (std base64 of <PLUGIN> XML) down to the
    Surge patch stream."""
    xml_bytes = base64.b64decode(preset_value)
    root = ET.fromstring(xml_bytes.decode("utf-8", "replace"))
    if root.tag != "PLUGIN":
        raise ValueError(f"expected <PLUGIN> root, got <{root.tag}>")
    state_attr = root.get("state")
    if not state_attr:
        raise ValueError("<PLUGIN> has no state attribute")
    attrs = {k: v for k, v in root.attrib.items() if k != "state"}
    raw_state = juce_b64.decode(state_attr)
    return Wrapper(chunk=unwrap_vst3_state(raw_state), attrs=attrs, raw_state=raw_state)


def encode_wrapper(chunk: bytes, attrs: Optional[Dict[str, str]] = None) -> str:
    """Surge patch stream → the standard-base64 <PLUGIN> XML string the S&S client
    applies verbatim via engine.setPluginState (same shape as bundled preset values)."""
    merged = dict(DEFAULT_PLUGIN_ATTRS)
    if attrs:
        merged.update(attrs)
    state_attr = juce_b64.encode(wrap_vst3_state(chunk))
    el = ET.Element("PLUGIN", {**merged, "state": state_attr})
    xml_text = _XML_DECL + "\n\n" + ET.tostring(el, encoding="unicode")
    return base64.b64encode(xml_text.encode("utf-8")).decode("ascii")


# ---- Layer 1: Surge native patch stream ('sub3') ------------------------------

_SUB3 = b"sub3"


@dataclass
class PatchStream:
    xml: bytes                # the <patch revision=N> XML document
    has_wavetables: bool      # any embedded wavetable/sample data after the XML
    raw: bytes                # the full stream as given

    @property
    def meta(self) -> Dict[str, str]:
        m = re.search(rb"<meta\b[^>]*>", self.xml)
        if not m:
            return {}
        frag = m.group(0)
        if not frag.endswith(b"/>"):
            frag = frag[:-1] + b"/>"
        try:
            return dict(ET.fromstring(frag.decode("utf-8", "replace")).attrib)
        except ET.ParseError:
            return {}

    @property
    def revision(self) -> Optional[int]:
        m = re.search(rb'<patch[^>]*\brevision="(\d+)"', self.xml)
        return int(m.group(1)) if m else None


def parse_patch_stream(stream: bytes) -> PatchStream:
    """Parse Surge's native patch stream: 'sub3' + xmlsize(i32 LE) + wtsize[2][3]
    (i32 LE) + XML + embedded wavetable blocks. Falls back to scanning if needed."""
    if stream[:4] != _SUB3:
        idx = stream.find(_SUB3)
        if idx < 0:
            raise ValueError("no 'sub3' magic in patch stream")
        stream = stream[idx:]
    xmlsize = int.from_bytes(stream[4:8], "little", signed=True)
    xml_start = stream.find(b"<?xml")
    if xml_start < 0 or xml_start > 4096:
        raise ValueError("no XML document found in patch stream")
    if not (0 < xmlsize <= len(stream) - xml_start):
        xmlsize = len(stream) - xml_start  # defensive: treat remainder as XML
    xml = stream[xml_start:xml_start + xmlsize]
    trailing = stream[xml_start + xmlsize:]
    return PatchStream(xml=xml, has_wavetables=len(trailing.strip(b"\x00")) > 0, raw=stream)
