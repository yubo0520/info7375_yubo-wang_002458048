from __future__ import annotations

import argparse
import json
import os


CONFIGS = {
    "BASELINE": os.path.join(os.path.dirname(__file__), "step3_baselines.json"),
}


def _jload(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def run(cur_p: str, base_p: str, out_p: str):
    cur = _jload(cur_p)
    base = _jload(base_p)
    rows = []
    out = {
        "current": cur_p,
        "baseline": base_p,
        "rows": rows,
    }

    for name, meta in cur.get("benchmarks", {}).items():
        cur_v = meta.get("main_metric")
        base_v = base.get(name)
        delta = None if base_v is None else cur_v - float(base_v)
        rows.append(
            {
                "benchmark": name,
                "current": cur_v,
                "step3": base_v,
                "delta": delta,
                "n": meta.get("n"),
                "kind": meta.get("kind"),
            }
        )

    os.makedirs(os.path.dirname(out_p) or ".", exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("benchmark,current,step3,delta,n,kind")
    for x in rows:
        print(
            f"{x['benchmark']},{x['current']},{x['step3']},{x['delta']},{x['n']},{x['kind']}"
        )
    print(f"[ok] compare -> {out_p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True)
    ap.add_argument("--baseline", default=CONFIGS["BASELINE"])
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    run(a.current, a.baseline, a.output)
