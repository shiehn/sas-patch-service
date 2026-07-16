"""GATE-1 blind listening A/B: semantic retrieval vs today's random-within-category.

Single-file, stdlib-only web app (the CLAP query embedding runs once at startup).

  .venv/bin/python eval/listen_ab.py            # http://localhost:8765
  .venv/bin/python eval/listen_ab.py --report   # tally votes and exit

Per golden query you hear two blinded sides:
  side X: top-5 OPEN semantic retrieval (no category mask, pooled+best-probe rerank)
  side Y: 5 uniform-random patches from the role-mapped category — i.e. exactly what
          S&S ships today (LLM picks category, then random)
Sides are shuffled per query with a seeded RNG; the mapping is stored server-side
only. Votes append to data/ab_votes.jsonl. GATE 1 target: semantic preferred on
≥70% of decided queries.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from sps import DATA_DIR, REPO_ROOT

INDEX = DATA_DIR / os.environ.get("SPS_INDEX_DIR", "index")
VOTES = DATA_DIR / "ab_votes.jsonl"
TEMPLATE = os.environ.get("SPS_TEMPLATE", "this is the sound of {q}")

# role → category substrings used ONLY to emulate today's baseline behavior
ROLE_TO_CATEGORY = {
    "bass": ["Bass"], "lead": ["Lead"], "pad": ["Pad"], "keys": ["Keys", "Piano", "Organ"],
    "pluck": ["Pluck"], "strings": ["String"], "brass": ["Brass"],
    "percussion": ["Percussion", "Drum", "Kick", "Snare", "Hat"],
    "texture": ["Atmosphere", "Drone", "FX", "Ambiance"], "fx": ["FX"],
}


def safe_id(pid: str) -> str:
    return pid.replace("/", "__").replace(" ", "_")


def build_pairs(mode: str = "gate1", anchors: str = ""):
    if mode == "gate2":
        coverage = json.loads((DATA_DIR / "anchor_coverage.json").read_text())
        have_gen = set()
        for l in (DATA_DIR / "corpus.jsonl").read_text().splitlines():
            r = json.loads(l)
            if r.get("source") == "generated" and "campaign" in r:
                have_gen.add(r["campaign"].get("anchor_id"))
        golden = [{"id": a["id"], "text": a["text"], "role": a["role"]}
                  for a in coverage["anchors"] if a["id"] in have_gen]
        if anchors:
            wanted = set(anchors.split(","))
            golden = [g for g in golden if g["id"] in wanted]
        random.Random(20260716).shuffle(golden)
        golden = golden[:40]
    else:
        golden = json.loads((REPO_ROOT / "eval" / "golden_queries.json").read_text())["queries"]
    pooled = np.load(INDEX / "pooled.npy")
    rows = [json.loads(l) for l in (INDEX / "pooled.jsonl").read_text().splitlines()]
    obs = np.load(INDEX / "obs.npy")
    obs_rows = [json.loads(l) for l in (INDEX / "obs.jsonl").read_text().splitlines()]

    from sps.clap_embed import ClapEmbedder
    embedder = ClapEmbedder(device="cpu")
    q_vecs = embedder.embed_text([TEMPLATE.format(q=q["text"]) for q in golden])

    def entry(r, best_probe):
        d = DATA_DIR / "renders" / safe_id(r["id"])
        flacs = sorted(p.name for p in d.glob("*.flac")) if d.is_dir() else []
        pick = f"{best_probe}.flac" if best_probe and f"{best_probe}.flac" in flacs else (flacs[0] if flacs else None)
        return {"id": r["id"], "name": r["name"], "category": r["category"],
                "audio": f"/audio/{safe_id(r['id'])}/{pick}" if pick else None}

    rng = random.Random(20260715)
    pairs = []
    for qi, (q, qv) in enumerate(zip(golden, q_vecs)):
        sims = pooled @ qv
        cand = np.argsort(-sims)[:200]
        scored = []
        for p in cand:
            r = rows[p]
            probe_scores = {obs_rows[i]["probe"]: float(obs[i] @ qv) for i in r["obs_idx"]}
            best_probe = max(probe_scores, key=probe_scores.get)
            scored.append((0.5 * float(sims[p]) + 0.5 * probe_scores[best_probe], r, best_probe))
        scored.sort(key=lambda t: -t[0])
        semantic = [entry(r, bp) for _, r, bp in scored[:5]]

        if mode == "gate2":
            # generated pool vs human pool — each side is its pool's SEMANTIC best;
            # in tallies, 'semantic' = the generated side, 'baseline' = human side
            gen = [(sc, r, bp) for sc, r, bp in scored if r["source"] == "generated"][:5]
            hum = [(sc, r, bp) for sc, r, bp in scored if r["source"] != "generated"][:5]
            semantic = [entry(r, bp) for _, r, bp in gen]
            baseline = [entry(r, bp) for _, r, bp in hum]
            if not semantic or not baseline:
                continue  # this query can't field both pools — skip the pair
        else:
            subs = ROLE_TO_CATEGORY.get(q.get("role", ""), [])
            pool = [r for r in rows if any(s.lower() in r["category"].lower() for s in subs)] or rows
            baseline = [entry(r, None) for r in rng.sample(pool, min(5, len(pool)))]

        flip = rng.random() < 0.5
        pairs.append({
            "qi": len(pairs), "id": q["id"], "text": q["text"], "role": q.get("role", ""),
            "sideA": baseline if flip else semantic,
            "sideB": semantic if flip else baseline,
            "a_is": "baseline" if flip else "semantic",
            "mode": mode,
        })
    return pairs


PAGE = """<!doctype html><meta charset="utf-8"><title>Blind A/B</title>
<style>
 body{font:15px -apple-system,sans-serif;margin:2rem auto;max-width:960px;padding:0 1rem;background:#111;color:#eee}
 h2{font-weight:600} .q{color:#7fd4ff;font-size:1.25rem}
 .cols{display:flex;gap:2rem} .col{flex:1;background:#1b1b1b;border-radius:10px;padding:1rem}
 .col h3{margin-top:0} audio{width:100%;margin:.3rem 0}
 button{font-size:1rem;padding:.55rem 1.2rem;margin:.4rem .4rem 0 0;border-radius:8px;border:0;cursor:pointer;background:#2d6cdf;color:#fff}
 button.alt{background:#444} .tally{color:#9f9;margin-left:1rem} .idx{color:#888}
</style>
<h2>Blind A/B — which set fits the description better? <span class="tally" id="tally"></span></h2>
<div class="q" id="qtext"></div><div class="idx" id="idx"></div>
<div class="cols">
 <div class="col"><h3>Side A</h3><div id="a"></div></div>
 <div class="col"><h3>Side B</h3><div id="b"></div></div>
</div>
<div>
 <button onclick="vote('A')">A wins</button><button onclick="vote('B')">B wins</button>
 <button class="alt" onclick="vote('tie')">Tie</button>
 <button class="alt" onclick="load(cur+1)">Skip →</button>
</div>
<script>
let cur=0,total=0;
function render(el,items){el.innerHTML=items.map((x,i)=>`<div>${i+1}. <audio controls preload="none" src="${x.audio||''}"></audio></div>`).join('')}
async function load(i){const r=await fetch('/api/pair?i='+i);if(!r.ok){document.body.innerHTML='<h2>done — run with --report</h2>';return}
 const p=await r.json();cur=p.qi;total=p.total;
 document.getElementById('qtext').textContent='"'+p.text+'"';
 document.getElementById('idx').textContent=`query ${p.qi+1}/${p.total} · ${p.id} · role: ${p.role||'—'}`;
 render(document.getElementById('a'),p.sideA);render(document.getElementById('b'),p.sideB);
 const t=await (await fetch('/api/tally')).json();
 document.getElementById('tally').textContent=`semantic ${t.semantic} · baseline ${t.baseline} · tie ${t.tie}`}
async function vote(v){await fetch('/api/vote',{method:'POST',body:JSON.stringify({qi:cur,vote:v})});load(cur+1)}
load(0);
</script>"""


class Handler(BaseHTTPRequestHandler):
    pairs = []

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif u.path == "/api/pair":
            i = int(parse_qs(u.query).get("i", ["0"])[0])
            if i >= len(self.pairs):
                return self._json({"done": True}, 404)
            p = dict(self.pairs[i])
            p.pop("a_is")  # never leak the blinding to the client
            p["total"] = len(self.pairs)
            self._json(p)
        elif u.path == "/api/tally":
            self._json(tally())
        elif u.path.startswith("/audio/"):
            rel = u.path[len("/audio/"):]
            f = (DATA_DIR / "renders" / rel).resolve()
            if not str(f).startswith(str((DATA_DIR / "renders").resolve())) or not f.is_file():
                return self._json({"error": "not found"}, 404)
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/flac")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/vote":
            return self._json({"error": "not found"}, 404)
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        p = self.pairs[payload["qi"]]
        v = payload["vote"]
        winner = "tie" if v == "tie" else (p["a_is"] if v == "A" else
                                           ("semantic" if p["a_is"] == "baseline" else "baseline"))
        with open(VOTES, "a") as f:
            f.write(json.dumps({"query_id": p["id"], "vote_side": v, "winner": winner,
                                "mode": p.get("mode", "gate1")}) + "\n")
        self._json({"ok": True})

    def log_message(self, *a):  # quiet
        pass


def tally(mode: str = None):
    latest = {}
    if VOTES.exists():
        for l in VOTES.read_text().splitlines():
            row = json.loads(l)
            row_mode = row.get("mode", "gate1")
            if mode and row_mode != mode:
                continue
            latest[(row_mode, row["query_id"])] = row["winner"]  # last vote wins
    t = {"semantic": 0, "baseline": 0, "tie": 0}
    for winner in latest.values():
        t[winner] += 1
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--anchors", default="", help="gate2: restrict to these anchor ids")
    ap.add_argument("--mode", default="gate1", choices=["gate1", "gate2"],
                    help="gate1: semantic vs random-category. gate2: generated vs human "
                         "(both semantically retrieved; 'semantic' in tallies = generated side)")
    args = ap.parse_args()

    if args.report:
        for mode, label, target in (("gate1", "GATE 1 semantic-vs-random (≥70%)", 0.70),
                                    ("gate2", "GATE 2 generated-vs-human (≥40%)", 0.40)):
            t = tally(mode)
            decided = t["semantic"] + t["baseline"]
            if decided + t["tie"] == 0:
                continue
            rate = (t["semantic"] / decided) if decided else 0.0
            print(f"{label}: {t}  win-rate {rate:.1%} → {'PASS' if rate >= target else 'FAIL'}")
        return

    print(f"building pairs (mode={args.mode})...")
    Handler.pairs = build_pairs(args.mode, anchors=args.anchors)
    print(f"ready: {len(Handler.pairs)} queries → http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
