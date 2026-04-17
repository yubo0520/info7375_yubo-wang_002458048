"""
AgentFlow Step 5/6: GRPO + LoRA on Qwen3.5-0.8B
TRL route (not veRL) — single L40S, much simpler.
"""

import os, re, json, glob, argparse
import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback
from peft import LoraConfig
try:
    from trl import GRPOConfig, GRPOTrainer
    from trl.rewards import think_format_reward
except Exception as e:
    raise RuntimeError(
        "Failed to import GRPO from `trl`. "
        "Please use a compatible stack (e.g., torch>=2.4 and a recent trl with GRPO support), "
        "or run this script in the provided Linux/Modal environment."
    ) from e

# ── config ───────────────────────────────────────────────────────────────────
CONFIGS = {
    "model":       "Qwen/Qwen3.5-0.8B-Instruct",
    "lora_r":      16,
    "lora_alpha":  64,
    "lora_drop":   0.05,
    "lr":          5e-6,
    "epochs":      1,
    "batch":       4,
    "grad_accum":  2,
    "n_gen":       4,
    "max_comp":    512,
    "beta":        0.001,
    "log_steps":   5,
    "save_steps":  50,
    "reward_w":    [2.0, 1.0],   # accuracy 2x, format 1x
}

# benchmarks used for training data (search-intensive tasks)
TRAIN_BENCHMARKS = ["bamboogle", "2wiki", "hotpotqa", "musique"]

# Task 6: BIRD Text-to-SQL
BIRD_PROMPT = (
    "You are an expert SQL assistant. "
    "Think step by step in <think></think> tags. "
    "Output ONLY the final SQL in <answer></answer> tags."
)


# ── data ─────────────────────────────────────────────────────────────────────

def mk_prompt(question):
    # strip embedded instructions if present, keep only the question
    q = re.split(r'When ready,|Output the final', question)[0].strip()
    return [
        {"role": "system",
         "content": "You are a helpful assistant. Think carefully before answering."},
        {"role": "user",
         "content": (
             f"{q}\n"
             "Show your reasoning in <think></think> tags. "
             "Give the final answer in <answer></answer> tags."
         )},
    ]


def _load_bench(data_dir, benchmarks, max_per=None):
    records = []
    for b in benchmarks:
        path = os.path.join(data_dir, b, "data", "data.json")
        if not os.path.exists(path):
            print(f"[data] missing: {path}, skip")
            continue
        with open(path) as f:
            items = json.load(f)
        if max_per:
            items = items[:max_per]
        for it in items:
            q = it.get("query") or it.get("question", "")
            ans = it.get("answer", "")
            records.append({
                "prompt": mk_prompt(q),
                "answer": ans if isinstance(ans, list) else [ans],
            })
        print(f"[data] {b}: {len(items)} samples")
    return Dataset.from_list(records)


# ── bird data (task 6) ───────────────────────────────────────────────────────

def _load_bird(n=None):
    from datasets import load_dataset
    ds = load_dataset("xlangai/bird", split="train", trust_remote_code=True)
    if n:
        ds = ds.select(range(min(n, len(ds))))
    records = []
    for row in ds:
        q    = row["question"]
        hint = row.get("evidence", "")
        db   = row.get("db_id", "")
        user = f"Database: {db}\nQuestion: {q}"
        if hint:
            user += f"\nHint: {hint}"
        user += "\nGenerate the SQL query."
        records.append({
            "prompt": [
                {"role": "system", "content": BIRD_PROMPT},
                {"role": "user",   "content": user},
            ],
            "answer": [row.get("SQL", "")],
        })
    print(f"[bird] {len(records)} samples")
    return Dataset.from_list(records)


def _sql_reward(completions, answer, **kwargs):
    # keyword-based fuzzy match for SQL
    import re
    rewards = []
    for comp, ans_list in zip(completions, answer):
        text = comp[-1]["content"] if isinstance(comp, list) else comp
        pred = _extract_ans(text)
        if pred is None:
            rewards.append(0.0)
            continue
        gold = ans_list[0] if ans_list else ""
        pred_l, gold_l = pred.lower(), gold.lower()
        kws = re.findall(r'\b(select|from|where|group|order|join|having)\b', gold_l)
        hit = sum(1 for k in kws if k in pred_l) / max(len(kws), 1)
        rewards.append(hit)
    return rewards


# ── reward ────────────────────────────────────────────────────────────────────

def _extract_ans(text):
    m = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return m[-1].strip().lower() if m else None


def acc_reward(completions, answer, **kwargs):
    rewards = []
    for comp, ans_list in zip(completions, answer):
        text = comp[-1]["content"] if isinstance(comp, list) else comp
        pred = _extract_ans(text)
        if pred is None:
            rewards.append(0.0)
            continue
        # fuzzy matching — any gold answer substring
        hit = any(a.lower() in pred or pred in a.lower() for a in ans_list)
        rewards.append(1.0 if hit else 0.0)
    return rewards


# ── callback ──────────────────────────────────────────────────────────────────

class ShowCb(TrainerCallback):
    def __init__(self, tok, samples, every=25):
        self.tok = tok
        self.samples = samples
        self.every = every

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.every != 0:
            return
        if model is None:
            return
        model.eval()
        print(f"\n--- step {state.global_step} examples ---")
        for ex in self.samples[:2]:
            ids = self.tok.apply_chat_template(
                ex["prompt"], tokenize=True,
                add_generation_prompt=True, return_tensors="pt"
            )["input_ids"].to(model.device)
            mask = torch.ones_like(ids)
            with torch.no_grad():
                out = model.generate(
                    input_ids=ids, attention_mask=mask,
                    max_new_tokens=256, do_sample=False,
                    pad_token_id=self.tok.eos_token_id
                )
            gen = self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            pred = _extract_ans(gen)
            hit = any(a.lower() in (pred or "") for a in ex["answer"])
            print(f"  ans={ex['answer']}  pred={pred}  ok={hit}")
            print(f"  output: {gen[:200]}")
        model.train()


# ── main ──────────────────────────────────────────────────────────────────────

def train(data_dir, out_dir, dev=False, task="step5"):
    # task="step6" → use BIRD dataset instead of benchmarks
    cfg = CONFIGS.copy()
    model_name = cfg["model"]

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # dev mode: tiny run to check for OOM/errors
    max_per = 10 if dev else None
    if task == "step6":
        ds = _load_bird(n=max_per)
        reward_fns = [_sql_reward, think_format_reward]
    else:
        ds = _load_bench(data_dir, TRAIN_BENCHMARKS, max_per=max_per)
        reward_fns = [acc_reward, think_format_reward]
    print(f"[data] total: {len(ds)}")

    if dev:
        cfg.update({"batch": 2, "grad_accum": 1, "n_gen": 2,
                    "epochs": 1, "save_steps": 5, "log_steps": 1})
        print("[dev] overrides active")

    peft_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_drop"],
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    grpo_cfg = GRPOConfig(
        output_dir=out_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        num_generations=cfg["n_gen"],
        max_completion_length=cfg["max_comp"],
        beta=cfg["beta"],
        logging_steps=cfg["log_steps"],
        bf16=True,
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        report_to="none",
        reward_weights=cfg["reward_w"],
    )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_fns,
        args=grpo_cfg,
        train_dataset=ds,
        peft_config=peft_cfg,
    )

    eval_samples = list(ds.select(range(min(3, len(ds)))))
    trainer.add_callback(ShowCb(tok, eval_samples, every=25))

    has_ckpt = os.path.isdir(out_dir) and any(
        d.startswith("checkpoint-") for d in os.listdir(out_dir)
    )
    trainer.train(resume_from_checkpoint=has_ckpt)

    print(f"[done] checkpoints → {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="../AgentFlow/test")
    p.add_argument("--out_dir", default="./local_ckpt")
    p.add_argument("--task", choices=["step5", "step6"], default="step5")
    p.add_argument("--dev", action="store_true", help="Run tiny debug training")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        dev=args.dev,
        task=args.task,
    )
