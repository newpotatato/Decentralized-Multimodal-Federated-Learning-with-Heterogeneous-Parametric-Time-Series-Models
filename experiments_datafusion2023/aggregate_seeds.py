"""Aggregate fig3 JSON results from multiple seeds into one file for analyze_and_plot.py."""
import json
import sys
from pathlib import Path

artifacts_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])

seeds = [42, 52, 62]
merged = []
for seed in seeds:
    f = artifacts_dir / f"seed{seed}" / "fig3_decentralized_methods.json"
    if not f.exists():
        print(f"WARNING: {f} not found")
        continue
    data = json.loads(f.read_text(encoding="utf-8"))
    for r in data.get("results", []):
        r["base_seed"] = seed
        if r.get("rounds") is None and r.get("history"):
            r["rounds"] = len(r["history"])
    merged.extend(data.get("results", []))

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps({"results": merged}, indent=2), encoding="utf-8")
print(f"Aggregated {len(merged)} experiments from {len(seeds)} seeds -> {out_path}")
