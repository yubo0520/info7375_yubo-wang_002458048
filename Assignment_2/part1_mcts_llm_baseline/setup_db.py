import sqlite3
import json
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_DIR, "misq_hf.db")
DATA_DIR = os.path.join(_DIR, "data")


def make_tables(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            self_repo TEXT,
            target TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS possibilities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT NOT NULL,
            item TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiment_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER NOT NULL,
            dataset_name TEXT NOT NULL,
            method TEXT NOT NULL,
            model TEXT NOT NULL,
            omega_aware INTEGER NOT NULL DEFAULT 1,
            success INTEGER NOT NULL,
            num_turns INTEGER NOT NULL,
            qgc REAL NOT NULL DEFAULT 0,
            conversation_log TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (sample_id) REFERENCES samples(id)
        )
    """)


def _load_dataset(cur, ds_name, filepath):
    if not os.path.exists(filepath):
        print(f"  [skip] {filepath} not found")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    targets = set()
    for i, item in enumerate(data):
        sr = item.get("self_repo", "")
        tgt = item.get("target", "")
        cur.execute(
            "INSERT INTO samples (dataset_name, sample_index, self_repo, target) VALUES (?, ?, ?, ?)",
            (ds_name, i, sr, tgt)
        )
        targets.add(tgt)

    for t in sorted(targets):
        cur.execute("INSERT INTO possibilities (dataset_name, item) VALUES (?, ?)", (ds_name, t))
    print(f"  [{ds_name}] {len(data)} samples, |O| = {len(targets)}")
    return sorted(targets)


def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed old db: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("1. Creating tables...")
    make_tables(cur)

    print("\n2. Loading data...")

    # DX
    omega = _load_dataset(cur, "DX", os.path.join(DATA_DIR, "DX.json"))
    if omega:
        print(f"  [DX] O = {omega}")

    conn.commit()

    # check
    print("\n3. Checking...")
    cur.execute("SELECT dataset_name, COUNT(*) FROM samples GROUP BY dataset_name")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]} samples")
    cur.execute("SELECT dataset_name, COUNT(*) FROM possibilities GROUP BY dataset_name")
    for row in cur.fetchall():
        print(f"  {row[0]}: |O| = {row[1]}")

    # check examples
    print("\n--- DX examples ---")
    cur.execute("SELECT sample_index, self_repo, target FROM samples WHERE dataset_name='DX' LIMIT 3")
    for row in cur.fetchall():
        print(f"  #{row[0]}: target={row[2]}, self_repo={row[1][:80]}...")

    conn.close()
    print(f"\nDone! DB: {DB_PATH}")


if __name__ == "__main__":
    main()
