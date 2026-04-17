from __future__ import annotations

import os
from pathlib import Path
import modal

_HERE = Path(__file__).parent  # task56_iso_new/


CONFIGS = {
    "APP": "task56-agentflow-dev",
    "HF_VOL": "task56-dev-hf",
    "RUN_VOL": "task56-dev-runs",
    "GPU": "A10G",
}

app = modal.App(CONFIGS["APP"])
hf = modal.Volume.from_name(CONFIGS["HF_VOL"], create_if_missing=True)
runs = modal.Volume.from_name(CONFIGS["RUN_VOL"], create_if_missing=True)

img = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12")
    .env({"PYTHONUNBUFFERED": "1"})  # SERPER_API_KEY injected at runtime via secret, not baked into image
    .apt_install(["git"])
    .pip_install(
        [
            "torch==2.6.0",
            "torchvision",
            "pillow",
            "transformers>=4.52.0",
            "datasets==3.2.0",
            "peft==0.14.0",
            "trl>=0.18.1",
            "accelerate==1.4.0",
            "pandas==2.2.3",
            "pyarrow==19.0.1",
        ]
    )
    .add_local_dir(str(_HERE), remote_path="/root/task56_iso_new", ignore=["__pycache__", "*.pyc"])
)

# Inject SERPER_API_KEY at runtime via Secret.from_dict so it's available in the
# container as an env var. Falls back gracefully (wiki search) if not set locally.
_serper_key = os.environ.get("SERPER_API_KEY", "")
_secrets = [modal.Secret.from_dict({"SERPER_API_KEY": _serper_key})] if _serper_key else []


def _cmd(args, timeout=520):
    import subprocess
    import sys
    import time

    n = 2
    i = 0
    while i < n:
        i += 1
        print("[cmd]", " ".join([sys.executable] + args), f"(try={i}/{n})", flush=True)
        # Inherit container stdout/stderr directly — no PIPE, so forked DataLoader
        # workers cannot cause an EOF deadlock. Timeout kills a truly hung process.
        p = subprocess.Popen([sys.executable] + args)
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            raise RuntimeError(f"cmd timed out after {timeout}s: {' '.join(args)}")
        if p.returncode == 0:
            print("[cmd] ok", flush=True)
            return
        print(f"[cmd] rc={p.returncode}", flush=True)
        if i < n:
            print(f"[cmd] retry in {i * 5}s", flush=True)
            time.sleep(i * 5)
    raise RuntimeError(f"cmd failed after {n} tries")


@app.function(
    image=img,
    gpu="T4",
    timeout=120,
)
def smoke():
    import importlib
    import torch

    pkgs = ["torch", "transformers", "datasets", "peft", "trl"]
    ver = {}
    for p in pkgs:
        m = importlib.import_module(p)
        ver[p] = getattr(m, "__version__", "unknown")

    print("cuda_available=", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu_name=", torch.cuda.get_device_name(0))
    print("versions=", ver)
    return {"ok": True}


@app.function(
    image=img,
    gpu=CONFIGS["GPU"],
    timeout=600,
    memory=49152,
    volumes={"/root/.cache/huggingface": hf, "/runs": runs},
    secrets=_secrets,
)
def train_step5_dev():
    _cmd(["/root/task56_iso_new/train_dev.py", "--task", "step5", "--dev", "--retry", "2"])
    _cmd(["/root/task56_iso_new/eval_dev.py", "--summary", "/runs/task56_new/step5_dev/summary.json"])
    return {"ok": True, "task": "step5", "mode": "dev"}


@app.function(
    image=img,
    gpu=CONFIGS["GPU"],
    timeout=600,
    memory=49152,
    volumes={"/root/.cache/huggingface": hf, "/runs": runs},
    secrets=_secrets,
)
def train_step6_dev():
    _cmd(["/root/task56_iso_new/train_dev.py", "--task", "step6", "--dev", "--retry", "2"])
    _cmd(["/root/task56_iso_new/eval_dev.py", "--summary", "/runs/task56_new/step6_dev/summary.json"])
    return {"ok": True, "task": "step6", "mode": "dev"}


@app.function(
    image=img,
    gpu=CONFIGS["GPU"],
    timeout=7200,
    memory=65536,
    volumes={"/root/.cache/huggingface": hf, "/runs": runs},
    secrets=_secrets,
)
def train_step5_full():
    _cmd(
        [
            "/root/task56_iso_new/train_dev.py",
            "--task",
            "step5",
            "--train_size",
            "1000",
            "--max_steps",
            "300",
            "--retry",
            "2",
        ],
        timeout=7000,
    )
    _cmd(["/root/task56_iso_new/eval_dev.py", "--summary", "/runs/task56_new/step5_full/summary.json"])
    return {"ok": True, "task": "step5", "mode": "full"}


@app.function(
    image=img,
    gpu=CONFIGS["GPU"],
    timeout=7200,
    memory=65536,
    volumes={"/root/.cache/huggingface": hf, "/runs": runs},
    secrets=_secrets,
)
def train_step6_full():
    _cmd(
        [
            "/root/task56_iso_new/train_dev.py",
            "--task",
            "step6",
            "--train_size",
            "600",
            "--max_steps",
            "200",
            "--retry",
            "2",
        ],
        timeout=7000,
    )
    _cmd(["/root/task56_iso_new/eval_dev.py", "--summary", "/runs/task56_new/step6_full/summary.json"])
    return {"ok": True, "task": "step6", "mode": "full"}


@app.function(
    image=img,
    gpu=CONFIGS["GPU"],
    timeout=7200,
    memory=65536,
    volumes={"/root/.cache/huggingface": hf, "/runs": runs},
    secrets=_secrets,
)
def eval_step5_bench_dev():
    _cmd(
        [
            "/root/task56_iso_new/eval_bench.py",
            "--task",
            "step5",
            "--ckpt_dir",
            "/runs/task56_new/step5_dev",
            "--benchmarks",
            "bamboogle,2wiki,hotpotqa,musique,gaia",
            "--limit",
            "20",
            "--output_dir",
            "/runs/task56_new/bench_step5_dev",
        ],
        timeout=6000,
    )
    _cmd(
        [
            "/root/task56_iso_new/compare_step3.py",
            "--current",
            "/runs/task56_new/bench_step5_dev/final_scores.json",
            "--output",
            "/runs/task56_new/bench_step5_dev/compare_step3.json",
        ]
    )
    return {"ok": True, "task": "step5", "mode": "bench_dev"}


@app.function(
    image=img,
    gpu=CONFIGS["GPU"],
    timeout=7200,
    memory=65536,
    volumes={"/root/.cache/huggingface": hf, "/runs": runs},
    secrets=_secrets,
)
def eval_step6_bird_dev():
    _cmd(
        [
            "/root/task56_iso_new/eval_bench.py",
            "--task",
            "step6",
            "--ckpt_dir",
            "/runs/task56_new/step6_dev",
            "--benchmarks",
            "bird",
            "--limit",
            "50",
            "--output_dir",
            "/runs/task56_new/bench_step6_dev",
        ]
    )
    _cmd(
        [
            "/root/task56_iso_new/compare_step3.py",
            "--current",
            "/runs/task56_new/bench_step6_dev/final_scores.json",
            "--output",
            "/runs/task56_new/bench_step6_dev/compare_step3.json",
        ]
    )
    return {"ok": True, "task": "step6", "mode": "bench_dev"}

