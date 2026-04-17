"""
Scoring script for BIRD Text-to-SQL benchmark.
Uses execution accuracy: compare result sets of predicted SQL vs gold SQL.

Usage:
    python calculate_score_bird.py \
        --data_file bird/data/data.json \
        --result_dir bird/results/qwen3.5-2b \
        --bird_db_dir /path/to/dev_databases
"""

import argparse
import json
import os
import re
import sqlite3
from tqdm import tqdm


def extract_sql(text: str) -> str:
    """Extract SQL from <answer>...</answer> tags, or take full text."""
    if not text or not isinstance(text, str):
        return ""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: look for SELECT statement
    match = re.search(r"(SELECT\s+.*?)(?:;|$)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def execute_sql(db_path: str, sql: str):
    """Execute SQL and return result set as a frozenset of tuples."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return frozenset(tuple(str(v).strip().lower() if v is not None else "null" for v in row) for row in rows)
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--bird_db_dir", required=True)
    parser.add_argument("--output_file", default="bird_scores.json")
    args = parser.parse_args()

    with open(args.data_file) as f:
        data = json.load(f)
    data_by_pid = {str(item["pid"]): item for item in data}

    result_files = sorted(
        [f for f in os.listdir(args.result_dir) if f.startswith("output_") and f.endswith(".json")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    correct = 0
    total = 0
    errors = 0
    results = []

    for fname in tqdm(result_files, desc="Scoring"):
        pid = fname.replace("output_", "").replace(".json", "")
        with open(os.path.join(args.result_dir, fname)) as f:
            result = json.load(f)

        item = data_by_pid.get(pid)
        if not item:
            continue

        db_id = item.get("db_id", "")
        db_path = os.path.join(args.bird_db_dir, db_id, f"{db_id}.sqlite")
        gold_sql = item["answer"][0] if item.get("answer") else ""

        pred_text = result.get("direct_output", "")
        pred_sql = extract_sql(pred_text) if isinstance(pred_text, str) else ""

        gold_result = execute_sql(db_path, gold_sql)
        pred_result = execute_sql(db_path, pred_sql) if pred_sql else None

        if pred_result is None:
            is_correct = False
            errors += 1
        else:
            is_correct = (pred_result == gold_result)

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "pid": pid,
            "db_id": db_id,
            "question": item.get("original_question", ""),
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nExecution Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    print(f"SQL parse errors:   {errors}/{total}")

    output = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": errors,
        "results": results,
    }
    out_path = os.path.join(args.result_dir, args.output_file)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
