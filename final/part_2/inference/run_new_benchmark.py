"""
Task 4: AgentFlow on BIRD (Text-to-SQL) benchmark.
No Flow-GRPO training, inference only.

Usage:
    python run_new_benchmark.py --base_url http://10.253.196.237:8000/v1 \
        --model "lovedheart/Qwen3.5-9B-FP8" --n 50
"""

import os, re, json, argparse, time
from pathlib import Path
from datasets import load_dataset
from openai import OpenAI

# ── config ────────────────────────────────────────────────────────────────────
CONFIGS = {
    "max_tokens":  1024,
    "temperature": 0.0,
    "max_retry":   3,
    "sleep":       2,
}

SYS_PROMPT = (
    "You are an expert SQL assistant. "
    "Given a database question and optional hints, generate the correct SQL query. "
    "Think step by step in <think></think> tags. "
    "Output ONLY the final SQL in <answer></answer> tags, no explanation after."
)


# ── data ──────────────────────────────────────────────────────────────────────

def _load_bird(n=None, split="validation"):
    # xlangai/bird: fields = db_id, question, SQL, evidence, difficulty
    ds = load_dataset("xlangai/bird", split=split, trust_remote_code=True)
    if n:
        ds = ds.select(range(min(n, len(ds))))
    return ds


def mk_prompt(row):
    q     = row["question"]
    hint  = row.get("evidence", "")
    db    = row.get("db_id", "")
    parts = [f"Database: {db}", f"Question: {q}"]
    if hint:
        parts.append(f"Hint: {hint}")
    parts.append("Generate the SQL query to answer this question.")
    return "\n".join(parts)


# ── inference ─────────────────────────────────────────────────────────────────

def _call(client, model, prompt, cfg):
    for attempt in range(cfg["max_retry"]):
        try:
            rsp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",  "content": SYS_PROMPT},
                    {"role": "user",    "content": prompt},
                ],
                max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
            )
            return rsp.choices[0].message.content
        except:
            time.sleep(cfg["sleep"] * (attempt + 1))
    return None


def _extract_sql(text):
    if text is None:
        return None
    m = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if m:
        return m[-1].strip()
    # fallback: last code block
    m2 = re.findall(r'```(?:sql)?\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
    return m2[-1].strip() if m2 else text.strip()


# ── eval ──────────────────────────────────────────────────────────────────────

def _sql_match(pred, gold):
    if pred is None:
        return False
    # normalize whitespace + lowercase for loose match
    norm = lambda s: re.sub(r'\s+', ' ', s.lower().strip())
    return norm(pred) == norm(gold)


def _keyword_match(pred, gold):
    # fuzzy: check key clauses exist in pred
    if pred is None:
        return False
    pred_l = pred.lower()
    for kw in re.findall(r'\b(select|from|where|group|order|join|having)\b', gold.lower()):
        if kw not in pred_l:
            return False
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def run(args):
    client = OpenAI(
        api_key=args.api_key or "EMPTY",
        base_url=args.base_url,
    )
    cfg = CONFIGS.copy()

    print(f"[bird] loading dataset (n={args.n})...")
    ds = _load_bird(n=args.n)
    print(f"[bird] {len(ds)} samples, model={args.model}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    exact, fuzzy = 0, 0

    for i, row in enumerate(ds):
        out_file = out_dir / f"output_{i}.json"
        # skip completed
        if out_file.exists():
            saved = json.loads(out_file.read_text())
            results.append(saved)
            exact  += int(saved.get("exact_match", False))
            fuzzy  += int(saved.get("fuzzy_match", False))
            continue

        prompt = mk_prompt(row)
        raw    = _call(client, args.model, prompt, cfg)
        pred   = _extract_sql(raw)
        gold   = row.get("SQL", "")

        em = _sql_match(pred, gold)
        fm = _keyword_match(pred, gold)
        exact += int(em)
        fuzzy += int(fm)

        rec = {
            "idx":         i,
            "db_id":       row.get("db_id"),
            "question":    row["question"],
            "gold_sql":    gold,
            "pred_sql":    pred,
            "raw_output":  raw,
            "exact_match": em,
            "fuzzy_match": fm,
            "difficulty":  row.get("difficulty", ""),
        }
        out_file.write_text(json.dumps(rec, indent=2))
        results.append(rec)

        if i % 10 == 0:
            print(f"  [{i}/{len(ds)}] exact={exact} fuzzy={fuzzy}")

    total = len(results)
    summary = {
        "model":       args.model,
        "n":           total,
        "exact_match": exact / total if total else 0,
        "fuzzy_match": fuzzy / total if total else 0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] exact={summary['exact_match']:.1%}  fuzzy={summary['fuzzy_match']:.1%}")
    print(f"[saved] {out_dir}/")
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_url", default="http://10.253.196.237:8000/v1")
    p.add_argument("--model",    default="lovedheart/Qwen3.5-9B-FP8")
    p.add_argument("--api_key",  default=None)
    p.add_argument("--n",        type=int, default=None, help="sample size, None=all")
    p.add_argument("--out_dir",  default="results/bird")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
    # quick test: python run_new_benchmark.py --n 5
