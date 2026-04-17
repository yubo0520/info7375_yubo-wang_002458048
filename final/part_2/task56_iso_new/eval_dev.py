from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Path to summary.json")
    a = ap.parse_args()

    if not os.path.exists(a.summary):
        raise FileNotFoundError(a.summary)

    data = json.load(open(a.summary, "r", encoding="utf-8"))
    print("task:", data.get("task"))
    print("model:", data.get("model"))
    print("dev:", data.get("dev"))
    print("train_size:", data.get("train_size"))
    print("max_steps:", data.get("max_steps"))
    print("eval_n:", data.get("eval_n"))
    print("eval_acc:", data.get("eval_acc"))


if __name__ == "__main__":
    main()

