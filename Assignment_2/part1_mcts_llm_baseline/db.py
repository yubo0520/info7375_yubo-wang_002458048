import sqlite3
import json
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_DIR, "misq_hf.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def load_samples(ds_name, limit=None):
    # load samples
    conn = _conn()
    c = conn.cursor()
    sql = "SELECT id, sample_index, self_repo, target FROM samples WHERE dataset_name = ?"
    if limit:
        sql += f" LIMIT {limit}"
    c.execute(sql, (ds_name,))
    rows = c.fetchall()
    conn.close()
    return rows


def load_omega(ds_name):
    # load possibility space
    connection = sqlite3.connect(DB_PATH)
    c = connection.cursor()
    c.execute("SELECT item FROM possibilities WHERE dataset_name = ? ORDER BY item", (ds_name,))
    res = [r[0] for r in c.fetchall()]
    connection.close()
    return res


def save(sample_id, ds_name, method, model, omega_aware, success, n_turns, qgc, conv_log):
    conn = _conn()
    c = conn.cursor()
    c.execute(
        """INSERT INTO experiment_results
           (sample_id, dataset_name, method, model, omega_aware, success, num_turns, qgc, conversation_log)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (sample_id, ds_name, method, model, omega_aware, success, n_turns, qgc,
         json.dumps(conv_log, ensure_ascii=False))
    )
    conn.commit()
    conn.close()


def get_results(ds_name, method, model):
    conn = _conn()
    c = conn.cursor()
    c.execute(
        "SELECT success, num_turns, qgc FROM experiment_results WHERE dataset_name=? AND method=? AND model=?",
        (ds_name, method, model)
    )
    rows = c.fetchall()
    conn.close()
    return rows


def calc_metrics(ds_name, method, model):
    # calculate SR, MSC, Mean QGC
    rows = get_results(ds_name, method, model)
    if not rows:
        print(f"No results found: {ds_name}/{method}/{model}")
        return

    total = len(rows)
    succ = [r for r in rows if r[0] == 1]

    sr = len(succ) / total * 100
    msc = sum(r[1] for r in succ) / len(succ) if succ else 0
    avg_qgc = sum(r[2] for r in rows) / total

    print(f"\n=== {ds_name} | {method} | {model} ===")
    print(f"  SR:       {sr:.2f}% ({len(succ)}/{total})")
    print(f"  MSC:      {msc:.2f}")
    print(f"  Mean QGC: {avg_qgc:.2f}")
    return {"sr": sr, "msc": msc, "qgc": avg_qgc}
