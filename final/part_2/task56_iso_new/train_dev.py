from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from rollout_dev import reward_step5, reward_step6


CONFIGS = {
    "MODEL": "Qwen/Qwen3.5-0.8B",
    "OUT": {
        "step5": {"dev": "/runs/task56_new/step5_dev", "full": "/runs/task56_new/step5_full"},
        "step6": {"dev": "/runs/task56_new/step6_dev", "full": "/runs/task56_new/step6_full"},
    },
}


def _latest_ckpt(out_dir: str) -> str | None:
    if not os.path.isdir(out_dir):
        return None
    cks = []
    for x in os.listdir(out_dir):
        if x.startswith("checkpoint-"):
            try:
                step = int(x.split("-")[-1])
            except:
                step = -1
            cks.append((step, os.path.join(out_dir, x)))
    if not cks:
        return None
    cks.sort(key=lambda t: t[0])
    return cks[-1][1]


def _dump_error(out_dir: str, task: str, model: str, dev: bool, train_size: int, max_steps: int, err: str):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "summary_error.json")
    data = {
        "task": task,
        "model": model,
        "dev": dev,
        "train_size": train_size,
        "max_steps": max_steps,
        "error": err,
        "ts": int(time.time()),
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[err] summary -> {p}")


def _mk_step5(train_n: int, eval_n: int) -> tuple[Dataset, Dataset]:
    rows = []
    try:
        ds = load_dataset("hotpot_qa", "distractor", split="train")
        for x in ds:
            q = str(x.get("question", "")).strip()
            gt = str(x.get("answer", "")).strip()
            if q and gt:
                rows.append({"question": q, "ground_truth": gt})
            if len(rows) >= train_n + eval_n:
                break
    except:
        try:
            ds = load_dataset("nq_open", split="train")
            for x in ds:
                q = str(x.get("question", "")).strip()
                a = x.get("answer", [])
                gt = str(a[0] if isinstance(a, list) and a else "")
                if q and gt:
                    rows.append({"question": q, "ground_truth": gt})
                if len(rows) >= train_n + eval_n:
                    break
        except:
            pass

    if len(rows) < train_n + eval_n:
        seed = [
            ("What is the capital of France?", "Paris"),
            ("Who wrote Hamlet?", "William Shakespeare"),
            ("What is 2 + 2?", "4"),
            ("Largest planet in solar system?", "Jupiter"),
        ]
        while len(rows) < train_n + eval_n:
            q, a = seed[len(rows) % len(seed)]
            rows.append({"question": q, "ground_truth": a})

    def _fmt(ex):
        p = (
            "You are a search agent planner.\n"
            "Available tool: Serper_Search.\n"
            "If search is needed, write <search>your query</search> before the answer.\n"
            "Always end with final answer in <answer>...</answer>.\n\n"
            f"Question: {ex['question']}"
        )
        # Use multimodal content format so VLM chat templates (Qwen3.5) don't
        # iterate over the string character-by-character when looking for item["type"].
        return {"prompt": [{"role": "user", "content": [{"type": "text", "text": p}]}], "question": ex["question"], "ground_truth": ex["ground_truth"]}

    tr = Dataset.from_list(rows[:train_n]).map(_fmt, remove_columns=["question", "ground_truth"])
    va = Dataset.from_list(rows[train_n : train_n + eval_n]).map(_fmt, remove_columns=["question", "ground_truth"])
    return tr, va


def _mk_step6(train_n: int, eval_n: int) -> tuple[Dataset, Dataset]:
    # Try bundled BIRD dev set first, then fallback paths
    bird_paths = [
        "/root/task56_iso_new/bird_dev.json",  # bundled with code mount
        "/root/AgentFlow/test/bird/data/data.json",
    ]
    rows = []
    for bird in bird_paths:
        if not os.path.exists(bird):
            continue
        try:
            items = json.load(open(bird, "r", encoding="utf-8"))
            for x in items:
                q = str(x.get("question", x.get("query", ""))).strip()
                ans = x.get("SQL", x.get("answer", x.get("sql", x.get("gold_sql", x.get("query", "")))))
                if isinstance(ans, list):
                    ans = ans[0] if ans else ""
                gt = str(ans).strip()
                if q and gt:
                    rows.append({"question": q, "ground_truth": gt})
                if len(rows) >= train_n + eval_n:
                    break
            if rows:
                print(f"[data] loaded {len(rows)} BIRD rows from {bird}")
                break
        except Exception as e:
            print(f"[data] failed to load {bird}: {e}")
            continue

    if len(rows) < train_n + eval_n:
        seed = [
            ("Count all users in table users", "SELECT COUNT(*) FROM users"),
            ("List all city names from cities", "SELECT city_name FROM cities"),
            ("Count orders in orders table", "SELECT COUNT(*) FROM orders"),
        ]
        while len(rows) < train_n + eval_n:
            q, a = seed[len(rows) % len(seed)]
            rows.append({"question": q, "ground_truth": a})

    def _fmt(ex):
        p = (
            "You are a Text-to-SQL agent planner with tools SchemaLookup and SQLExecutor.\n"
            "Think short. You may draft SQL, then refine it.\n"
            "Return final SQL in <answer>...</answer>.\n\n"
            f"Question: {ex['question']}"
        )
        return {"prompt": [{"role": "user", "content": [{"type": "text", "text": p}]}], "question": ex["question"], "ground_truth": ex["ground_truth"]}

    tr = Dataset.from_list(rows[:train_n]).map(_fmt, remove_columns=["question", "ground_truth"])
    va = Dataset.from_list(rows[train_n : train_n + eval_n]).map(_fmt, remove_columns=["question", "ground_truth"])
    return tr, va


def _eval(model, tok, ds: Dataset, rw, n: int):
    model.eval()
    n = min(n, len(ds))
    ok = 0.0
    for i in range(n):
        ex = ds[i]
        p = tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
        inp = tok(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
        comp = tok.decode(out[0][inp["input_ids"].shape[1] :], skip_special_tokens=True)
        ok += rw([comp], [ex["ground_truth"]], [ex["question"]])[0]
    return ok / max(n, 1), n


def run(task: str, model: str, out_dir: str, train_size: int, max_steps: int, dev: bool):
    if dev:
        train_size = min(train_size, 80)
        max_steps = min(max_steps, 30)
        eval_n = 10
    else:
        eval_n = 50

    print(f"[run] task={task} dev={dev} train={train_size} steps={max_steps}")
    print(f"[run] cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[run] gpu={torch.cuda.get_device_name(0)}")

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if task == "step5":
        tr, va = _mk_step5(train_size, eval_n)
        rw = reward_step5
    else:
        tr, va = _mk_step6(train_size, eval_n)
        rw = reward_step6

    peft = LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    args = GRPOConfig(
        output_dir=out_dir,
        max_steps=max_steps,
        per_device_train_batch_size=2,   # must be divisible by num_generations (2)
        gradient_accumulation_steps=1 if dev else 2,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        num_generations=2 if dev else 4,
        max_completion_length=128 if dev else 256,
        beta=0.001,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not (torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False),
        logging_steps=5,
        save_strategy="steps",
        save_steps=max(10, max_steps // 3),
        save_total_limit=2,
        report_to="none",
        reward_weights=[1.0],
    )

    trn = GRPOTrainer(
        model=model,
        reward_funcs=[rw],
        args=args,
        train_dataset=tr,
        peft_config=peft,
    )
    ck = _latest_ckpt(out_dir)
    if ck:
        print(f"[run] resume checkpoint: {ck}")

    try:
        trn.train(resume_from_checkpoint=ck)
        acc, n_eval = _eval(trn.model, tok, va, rw, eval_n)
        print(f"[eval] acc={acc:.3f} n={n_eval}")
    except Exception as e:
        _dump_error(out_dir, task, model, dev, train_size, max_steps, str(e))
        raise

    os.makedirs(out_dir, exist_ok=True)
    summ = {
        "task": task,
        "model": model,
        "dev": dev,
        "train_size": train_size,
        "max_steps": max_steps,
        "eval_n": n_eval,
        "eval_acc": acc,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)
    print(f"[ok] summary -> {out_dir}/summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["step5", "step6"], required=True)
    ap.add_argument("--model", default=CONFIGS["MODEL"])
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--train_size", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--retry", type=int, default=1)
    a = ap.parse_args()

    mode = "dev" if a.dev else "full"
    out = a.output_dir or CONFIGS["OUT"][a.task][mode]
    i = 0
    while i < max(1, a.retry):
        try:
            run(a.task, a.model, out, a.train_size, a.max_steps, a.dev)
            break
        except Exception as e:
            i += 1
            if i >= max(1, a.retry):
                raise
            wait_s = i * 5
            print(f"[retry] fail={e} wait={wait_s}s")
            time.sleep(wait_s)

