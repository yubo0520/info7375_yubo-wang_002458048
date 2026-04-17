"""Generate report figures for final project part 2."""
import json, os
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "AgentFlow" / "test"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

QA_BENCHES = ["bamboogle", "2wiki", "hotpotqa", "musique", "gaia"]
QWEN35 = [
    ("qwen3.5-qwen3.5-0.8b", 0.8),
    ("qwen3.5-2b", 2),
    ("qwen3.5-4b", 4),
    ("qwen3.5-9b", 9),
    ("qwen3.5-27b", 27),
]


def read_qa(bench, label):
    f = TEST / bench / "results" / label / "final_scores_direct_output.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    for k in ("accuracy", "main_metric", "score"):
        if k in d:
            v = d[k]
            return v if v > 1 else v * 100
    return None


def read_bird(label):
    f = TEST / "bird" / "results" / label / "bird_scores.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return d.get("accuracy", 0) * 100


# --- Figure 1: Step 2 scaling curve ---
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.tab10.colors
for i, bench in enumerate(QA_BENCHES):
    xs, ys = [], []
    for label, size in QWEN35:
        y = read_qa(bench, label)
        if y is not None:
            xs.append(size)
            ys.append(y)
    ax.plot(xs, ys, marker="o", label=bench, color=colors[i])

ax.set_xscale("log")
ax.set_xticks([0.8, 2, 4, 9, 27])
ax.set_xticklabels(["0.8B", "2B", "4B", "9B", "27B"])
ax.set_xlabel("Model size (parameters)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Step 2 — Qwen3.5 scaling on AgentFlow QA benchmarks")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "step2_scaling.png", dpi=140)
plt.close()
print("wrote", OUT / "step2_scaling.png")


# --- Figure 2: Step 1 Qwen2.5-7B baseline per benchmark ---
fig, ax = plt.subplots(figsize=(7, 4.5))
vals = [read_qa(b, "qwen2.5-7b") for b in QA_BENCHES]
bars = ax.bar(QA_BENCHES, vals, color="#4c72b0")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.1f}%",
            ha="center", fontsize=9)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, max(vals) + 12)
ax.set_title("Step 1 — Qwen2.5-7B baseline on AgentFlow QA benchmarks (n=30)")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "step1_qwen25_7b.png", dpi=140)
plt.close()
print("wrote", OUT / "step1_qwen25_7b.png")


# --- Figure 3: BIRD scaling ---
fig, ax = plt.subplots(figsize=(7, 4.5))
bird_labels = [
    ("qwen3.5-0.8b", 0.8),
    ("qwen3.5-2b", 2),
    ("qwen3.5-4b", 4),
    ("qwen3.5-9b", 9),
    ("qwen3.5-27b", 27),
]
xs, ys = [], []
for label, size in bird_labels:
    v = read_bird(label)
    if v is not None:
        xs.append(size)
        ys.append(v)
ax.plot(xs, ys, marker="o", color="#c44e52", linewidth=2)
for x, y in zip(xs, ys):
    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                xytext=(6, 6), fontsize=9)
ax.set_xscale("log")
ax.set_xticks([0.8, 2, 4, 9, 27])
ax.set_xticklabels(["0.8B", "2B", "4B", "9B", "27B"])
ax.set_xlabel("Model size (parameters)")
ax.set_ylabel("Execution accuracy (%)")
ax.set_title("Step 3 — Qwen3.5 on BIRD Text-to-SQL (n=22–30)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "step3_bird_scaling.png", dpi=140)
plt.close()
print("wrote", OUT / "step3_bird_scaling.png")


# --- Figure 4: Training curves ---
ckpt_dir = ROOT / "train" / "local_ckpt" / "checkpoint-30"
hist = json.loads((ckpt_dir / "trainer_state.json").read_text())["log_history"]
steps = [h["step"] for h in hist]

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
metrics = [
    ("reward", "Mean reward"),
    ("loss", "Loss"),
    ("kl", "KL divergence"),
    ("entropy", "Entropy"),
]
for ax, (key, title) in zip(axes.flat, metrics):
    ys = [h.get(key, 0) for h in hist]
    ax.plot(steps, ys, marker="o", markersize=4)
    ax.set_xlabel("step")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
fig.suptitle("Step 4/5 — Flow-GRPO + LoRA training (Qwen3.5-0.8B, 30 steps)")
plt.tight_layout()
plt.savefig(OUT / "step45_training_curves.png", dpi=140)
plt.close()
print("wrote", OUT / "step45_training_curves.png")

print("done")
