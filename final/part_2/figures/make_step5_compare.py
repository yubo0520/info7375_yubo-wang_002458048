"""Step 5 eval vs Step 3 baseline bar chart (different agents)."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"

step3 = json.loads((ROOT / "task56_iso_new" / "step3_baselines.json").read_text())
step5 = json.loads((ROOT / "task56_iso_new" / "bench_step5_final_scores.json").read_text())
step6 = json.loads((ROOT / "task56_iso_new" / "bench_step6_final_scores.json").read_text())

benches = ["bamboogle", "2wiki", "hotpotqa", "musique", "gaia", "bird"]
step3_vals = [step3[b] * 100 for b in benches]
step5_vals = []
for b in benches:
    if b == "bird":
        step5_vals.append(step6["benchmarks"]["bird"]["main_metric"] * 100)
    else:
        step5_vals.append(step5["benchmarks"][b]["main_metric"] * 100)

x = np.arange(len(benches))
w = 0.35
fig, ax = plt.subplots(figsize=(9, 4.8))
b1 = ax.bar(x - w/2, step3_vals, w, label="Step 3 (full AgentFlow, 4 modules)", color="#4c72b0")
b2 = ax.bar(x + w/2, step5_vals, w, label="Step 4/5 (simplified agent, Flow-GRPO+LoRA)", color="#c44e52")
for bars, vals in [(b1, step3_vals), (b2, step5_vals)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.8, f"{v:.1f}",
                ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(benches)
ax.set_ylabel("Main metric (%)")
ax.set_title("Step 4/5 eval vs Step 3 baseline — Qwen3.5-0.8B\n(different agent architectures, not directly comparable)")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "step45_vs_step3.png", dpi=140)
print("wrote", OUT / "step45_vs_step3.png")
