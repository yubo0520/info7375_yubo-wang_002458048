import argparse
import sqlite3
import csv
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_DIR, "misq_hf.db")


def get_methods(ds, model=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if model:
        c.execute(
            "SELECT DISTINCT method, model FROM experiment_results WHERE dataset_name=? AND model=?",
            (ds, model))
    else:
        c.execute(
            "SELECT DISTINCT method, model FROM experiment_results WHERE dataset_name=?",
            (ds,))
    rows = c.fetchall()
    conn.close()
    return rows


def calc(ds, method, model):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT success, num_turns, qgc FROM experiment_results WHERE dataset_name=? AND method=? AND model=?",
        (ds, method, model))
    rows = c.fetchall()
    conn.close()

    if not rows:
        return None

    n = len(rows)
    succ = [r for r in rows if r[0] == 1]
    sr = len(succ) / n * 100
    msc = sum(r[1] for r in succ) / len(succ) if succ else 0
    avg_qgc = sum(r[2] for r in rows) / n

    return {
        "dataset": ds, "method": method, "model": model,
        "n_samples": n, "n_success": len(succ),
        "sr": sr, "msc": msc, "mean_qgc": avg_qgc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="DX")
    parser.add_argument("--model", default=None)
    parser.add_argument("--export", default=None)
    args = parser.parse_args()

    combos = get_methods(args.dataset, args.model)
    if not combos:
        print(f"No results for {args.dataset}.")
        print("Run: python run_baseline.py --method dp --dataset DX")
        return

    results = []
    for method, model in sorted(combos):
        m = calc(args.dataset, method, model)
        if m:
            results.append(m)

    print(f"\n{'='*60}")
    print(f"  Results: {args.dataset}")
    print(f"{'='*60}\n")

    print(f"{'Method':<10} {'Model':<20} {'SR':>8} {'MSC':>8} {'QGC':>8} {'N':>6}")
    print("-" * 60)
    for r in results:
        print(f"{r['method']:<10} {r['model']:<20} {r['sr']:>7.2f}% {r['msc']:>8.2f} {r['mean_qgc']:>8.2f} {r['n_success']}/{r['n_samples']:>3}")
    print()

    if args.export:
        with open(args.export, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=["dataset","method","model","n_samples","n_success","sr","msc","mean_qgc"])
            w.writeheader()
            w.writerows(results)
        print(f"Exported to: {args.export}")


if __name__ == "__main__":
    main()
