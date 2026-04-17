"""
Modal runner for Step 5 (paper benchmarks) & Step 6 (new benchmark).
Usage:
    modal run train/run_modal.py          # step5 full run
    modal run train/run_modal.py --dev    # 10-min dev check
    modal run train/run_modal.py --step 6 # step6 (new benchmark)
"""

import modal, subprocess, sys, os

# ── resources — prefix with your initials ────────────────────────────────────
app  = modal.App("yw-agentflow-train")
hf   = modal.Volume.from_name("yw-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("yw-runs",     create_if_missing=True)

# portkey secret for verifier/fixed model components
portkey_secret = modal.Secret.from_name("yw-portkey")

FLASH_WHL = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels"
    "/releases/download/v0.7.16"
    "/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl"
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env({"PYTHONUNBUFFERED": "1", "DEBIAN_FRONTEND": "noninteractive"})
    .apt_install(["git", "wget"])
    .pip_install([
        "torch==2.10.0",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "accelerate",
        "wandb",
    ])
    .pip_install(FLASH_WHL)
)


# ── training function ─────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="L40S",
    timeout=600,              # 10 min for dev; change to 7200 for full run
    volumes={
        "/root/.cache/huggingface": hf,
        "/runs": runs,
    },
    secrets=[portkey_secret],
    memory=65536,
)
def run_train(dev: bool = False, step: int = 5):
    import sys, os
    sys.path.insert(0, "/runs/code")

    runs.reload()

    code_dir = "/runs/code"
    data_dir = f"{code_dir}/AgentFlow/test"    # benchmark data
    out_dir  = f"/runs/step{step}/checkpoints"
    os.makedirs(out_dir, exist_ok=True)

    # install agentflow package (no deps to avoid version conflicts)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-q",
         "-e", f"{code_dir}/AgentFlow/agentflow/"],
        check=True,
    )

    from train_flow_grpo import train
    train(data_dir=data_dir, out_dir=out_dir, dev=dev, task=f"step{step}")

    runs.commit()
    print(f"[done] step{step} ckpts → {out_dir}")
    print(f"[download] modal volume get yw-runs /runs/step{step}/checkpoints/ ./step{step}_ckpts/")


# ── entrypoint ────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(dev: bool = False, step: int = 5):
    """
    --dev   : short test run (10 min timeout)
    --step  : 5 = paper benchmarks, 6 = new benchmark
    """
    print(f"[modal] step={step} dev={dev}")
    print("[tip] upload code first: modal volume put yw-runs . /code")
    run_train.remote(dev=dev, step=step)
