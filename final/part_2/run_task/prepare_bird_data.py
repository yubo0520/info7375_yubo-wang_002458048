"""
Prepare BIRD dev data for AgentFlow.

Usage:
    python prepare_bird_data.py \
        --bird_dev_json /path/to/bird/dev/dev.json \
        --bird_db_dir /path/to/bird/dev/dev_databases \
        --output_dir AgentFlow/test/bird/data \
        --n 30

BIRD dev.json can be downloaded from: https://bird-bench.github.io/
"""

import argparse
import json
import os
import random
import sqlite3


def get_schema(db_path: str) -> str:
    """Get CREATE TABLE statements from the database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL;")
        rows = cursor.fetchall()
        conn.close()
        return "\n".join(r[0] for r in rows if r[0])
    except Exception as e:
        return f"-- Schema unavailable: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bird_dev_json", required=True, help="Path to BIRD dev.json")
    parser.add_argument("--bird_db_dir", required=True, help="Path to dev_databases directory")
    parser.add_argument("--output_dir", default="AgentFlow/test/bird/data")
    parser.add_argument("--n", type=int, default=30, help="Number of questions to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.bird_dev_json, "r") as f:
        bird_data = json.load(f)

    print(f"Total BIRD dev questions: {len(bird_data)}")

    # Sample n questions, trying to cover diverse databases
    db_ids = list(set(item["db_id"] for item in bird_data))
    print(f"Total databases: {len(db_ids)}")

    # Sample proportionally from each db, up to n total
    selected = []
    per_db = max(1, args.n // len(db_ids))
    for db_id in sorted(db_ids):
        db_questions = [item for item in bird_data if item["db_id"] == db_id]
        selected.extend(random.sample(db_questions, min(per_db, len(db_questions))))

    # Trim or top-up to exactly n
    random.shuffle(selected)
    selected = selected[:args.n]
    print(f"Selected {len(selected)} questions from {len(set(s['db_id'] for s in selected))} databases")

    os.makedirs(args.output_dir, exist_ok=True)

    agentflow_data = []
    for idx, item in enumerate(selected):
        db_id = item["db_id"]
        db_path = os.path.join(args.bird_db_dir, db_id, f"{db_id}.sqlite")
        schema = get_schema(db_path)

        question_text = item["question"]
        evidence = item.get("evidence", "").strip()
        gold_sql = item.get("SQL", item.get("sql", "")).strip()

        # Build query for AgentFlow
        query = (
            f"You are an expert SQL assistant. Given the database schema and question, "
            f"generate the correct SQL query.\n\n"
            f"Database: {db_id}\n\n"
            f"Schema:\n{schema}\n\n"
            f"Question: {question_text}\n"
        )
        if evidence:
            query += f"Hint: {evidence}\n"
        query += (
            "\nGenerate a valid SQLite SQL query to answer the question. "
            "When ready, output ONLY the final SQL query enclosed in <answer> and </answer> tags. "
            "Do not generate any content after the </answer> tag."
        )

        entry = {
            "idx": idx,
            "question": query,
            "answer": [gold_sql],
            "gen_text_store": "",
            "pid": str(idx),
            "query": query,
            "image": None,
            "db_id": db_id,
            "original_question": question_text,
        }
        agentflow_data.append(entry)

    output_file = os.path.join(args.output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(agentflow_data, f, indent=2)

    print(f"Saved {len(agentflow_data)} entries to {output_file}")

    # Also save db_id mapping for scoring
    db_map = {str(item["pid"]): item["db_id"] for item in agentflow_data}
    with open(os.path.join(args.output_dir, "db_map.json"), "w") as f:
        json.dump(db_map, f, indent=2)
    print(f"Saved db_id mapping to {args.output_dir}/db_map.json")


if __name__ == "__main__":
    main()
