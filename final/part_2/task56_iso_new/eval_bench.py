from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from rollout_dev import reward_step5, reward_step6


CONFIGS = {
    "CFG": os.path.join(os.path.dirname(__file__), "bench_cfg.json"),
    "BASE": "Qwen/Qwen3.5-0.8B",
    "OUT": "/runs/task56_new/bench",
    "MAX_NEW": 192,
}


def _jload(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _ans(text: Any) -> str:
    if isinstance(text, list):
        text = text[-1]["content"] if text else ""
    txt = str(text)
    m = re.search(r"<answer>(.*?)</answer>", txt, re.DOTALL)
    return m.group(1).strip() if m else txt.strip()


def _gold(x: Any) -> list[str]:
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if x is None:
        return []
    return [str(x).strip()]


def _q(ex: dict[str, Any]) -> str:
    return str(ex.get("query", ex.get("question", ex.get("Question", "")))).strip()


def _cfg():
    return _jload(CONFIGS["CFG"])


def _latest_ckpt(root: str) -> str:
    if not root or not os.path.isdir(root):
        return root
    cks = []
    for x in os.listdir(root):
        if not x.startswith("checkpoint-"):
            continue
        try:
            step = int(x.split("-")[-1])
        except:
            step = -1
        cks.append((step, os.path.join(root, x)))
    if not cks:
        return root
    cks.sort(key=lambda t: t[0])
    return cks[-1][1]


def _p(task: str, q: str) -> list[dict[str, str]]:
    if task == "step6":
        txt = (
            "You are a Text-to-SQL agent planner with tools SchemaLookup and SQLExecutor.\n"
            "Think short. You may draft SQL, then refine it.\n"
            "Return final SQL in <answer>...</answer>.\n\n"
            f"Question: {q}"
        )
    else:
        txt = (
            "You are a search agent planner.\n"
            "Available tool: Serper_Search.\n"
            "If search is needed, write <search>your query</search> before the answer.\n"
            "Always end with final answer in <answer>...</answer>.\n\n"
            f"Question: {q}"
        )
    return [{"role": "user", "content": txt}]


def _load_model(model_name: str, ckpt_dir: str):
    ckpt_dir = _latest_ckpt(ckpt_dir)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if ckpt_dir and os.path.exists(os.path.join(ckpt_dir, "adapter_config.json")):
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None,
        )
        mdl = PeftModel.from_pretrained(base, ckpt_dir)
    else:
        use_dir = ckpt_dir if ckpt_dir else model_name
        mdl = AutoModelForCausalLM.from_pretrained(
            use_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None,
        )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(dev)
    mdl.eval()
    return mdl, tok


def _gen(mdl, tok, task: str, q: str) -> str:
    prompt = tok.apply_chat_template(_p(task, q), tokenize=False, add_generation_prompt=True)
    inp = tok(prompt, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(
            **inp,
            max_new_tokens=CONFIGS["MAX_NEW"],
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inp["input_ids"].shape[1] :], skip_special_tokens=True)


def _score(task: str, comp: str, golds: list[str], q: str) -> tuple[float, float]:
    pred = _norm(_ans(comp))
    ex = 0.0
    for g in golds:
        ng = _norm(g)
        if pred == ng or (pred and ng and (pred in ng or ng in pred)):
            ex = 1.0
            break
    if task == "step6":
        rw = reward_step6([comp], [golds[0] if golds else ""], [q])[0]
    else:
        rw = reward_step5([comp], [golds[0] if golds else ""], [q])[0]
    return ex, float(rw)


def _rows(name: str, cfg: dict[str, Any], limit: int):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), cfg[name]["data"]))
    rows = _jload(path)
    if limit > 0:
        rows = rows[:limit]
    return path, rows


def run(task: str, model_name: str, ckpt_dir: str, benches: list[str], out_dir: str, limit: int):
    os.makedirs(out_dir, exist_ok=True)
    cfg = _cfg()
    mdl, tok = _load_model(model_name, ckpt_dir)

    final_scores = {
        "task": task,
        "model": model_name,
        "ckpt_dir": ckpt_dir,
        "benchmarks": {},
    }

    for name in benches:
        meta = cfg[name]
        data_path, rows = _rows(name, cfg, limit)
        out = []
        ex_sum = 0.0
        rw_sum = 0.0
        print(f"[bench] {name} n={len(rows)} data={data_path}")
        for i, ex in enumerate(rows):
            q = _q(ex)
            golds = _gold(ex.get("answer", ex.get("Final answer", "")))
            comp = _gen(mdl, tok, task, q)
            exact, score = _score(task, comp, golds, q)
            ex_sum += exact
            rw_sum += score
            out.append(
                {
                    "i": i,
                    "pid": ex.get("pid", ex.get("idx", i)),
                    "query": q,
                    "gold": golds,
                    "pred_raw": comp,
                    "pred_answer": _ans(comp),
                    "exact": exact,
                    "score": score,
                }
            )
            if i % 10 == 0:
                print(f"[bench] {name} {i+1}/{len(rows)}")

        n = max(len(rows), 1)
        res = {
            "kind": meta["kind"],
            "n": len(rows),
            "exact_avg": ex_sum / n,
            "score_avg": rw_sum / n,
        }
        res["main_metric"] = res["score_avg"] if meta["kind"] == "sql" else res["exact_avg"]
        final_scores["benchmarks"][name] = res

        with open(os.path.join(out_dir, f"{name}_pred.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[bench] {name} main={res['main_metric']:.4f}")

    with open(os.path.join(out_dir, "final_scores.json"), "w", encoding="utf-8") as f:
        json.dump(final_scores, f, indent=2, ensure_ascii=False)
    print(f"[ok] final -> {out_dir}/final_scores.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["step5", "step6"], required=True)
    ap.add_argument("--model", default=CONFIGS["BASE"])
    ap.add_argument("--ckpt_dir", default="")
    ap.add_argument("--benchmarks", default="")
    ap.add_argument("--output_dir", default=CONFIGS["OUT"])
    ap.add_argument("--limit", type=int, default=20)
    a = ap.parse_args()

    cfg = _cfg()
    if a.benchmarks:
        benches = [x.strip() for x in a.benchmarks.split(",") if x.strip()]
    else:
        benches = [k for k, v in cfg.items() if v["task"] == a.task]
    run(a.task, a.model, a.ckpt_dir, benches, a.output_dir, a.limit)
