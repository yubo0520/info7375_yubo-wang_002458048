from __future__ import annotations

import json
import os
import re
import sqlite3
import urllib.parse
import urllib.request
from typing import Any


CONFIGS = {
    "SEARCH_LIMIT": 1500,
    "TOOL_TEXT": 200,
}


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _ans(text: Any) -> str:
    if isinstance(text, list):
        text = text[-1]["content"] if text else ""
    txt = str(text)
    m = re.search(r"<answer>(.*?)</answer>", txt, re.DOTALL)
    return m.group(1).strip() if m else txt.strip()


def _uniq(xs: list[str]) -> list[str]:
    out = []
    seen = set()
    for x in xs:
        k = _norm(x)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def _tabs(sql: str) -> list[str]:
    out = []
    for a, b in re.findall(r"\bfrom\s+([a-zA-Z_][\w]*)|\bjoin\s+([a-zA-Z_][\w]*)", sql, re.IGNORECASE):
        x = a or b
        if x:
            out.append(x)
    return _uniq(out)


def _cols(sql: str) -> list[str]:
    toks = re.findall(r"\b[a-zA-Z_][\w]*\b", str(sql))
    ban = {
        "select",
        "from",
        "where",
        "join",
        "on",
        "group",
        "by",
        "order",
        "limit",
        "count",
        "avg",
        "sum",
        "min",
        "max",
        "distinct",
        "as",
        "and",
        "or",
        "desc",
        "asc",
    }
    out = []
    for x in toks:
        if x.lower() in ban:
            continue
        if x.lower() in [t.lower() for t in _tabs(sql)]:
            continue
        out.append(x)
    base = ["id", "name", "value", "amount", "city_name", "column_name"]
    return _uniq(base + out)[:10]


def _col_tp(col: str) -> str:
    c = col.lower()
    if c in {"id", "value", "amount", "column_name", "age", "score", "count"}:
        return "INTEGER"
    return "TEXT"


def _conn(q: str, sql_a: str, sql_b: str):
    # rough sqlite sandbox, enough for reward
    db = sqlite3.connect(":memory:")
    cur = db.cursor()
    tabs = _tabs(f"{sql_a}\n{sql_b}") or ["table_name"]
    cols = _cols(f"{sql_a}\n{sql_b}\n{q}")
    defs = ", ".join([f'"{c}" {_col_tp(c)}' for c in cols])
    rows = [
        {c: (i + 1) * 10 if _col_tp(c) == "INTEGER" else f"{c}_{i+1}" for c in cols}
        for i in range(3)
    ]
    for t in tabs:
        try:
            cur.execute(f'CREATE TABLE "{t}" ({defs})')
            for r in rows:
                ks = ", ".join([f'"{k}"' for k in cols])
                vs = ", ".join(["?"] * len(cols))
                cur.execute(f'INSERT INTO "{t}" ({ks}) VALUES ({vs})', [r[k] for k in cols])
        except:
            pass
    db.commit()
    return db


def _sql_run(q: str, sql: str, gold: str) -> tuple[bool, str]:
    if not sql or "select" not in sql.lower():
        return False, "not_select"
    db = None
    try:
        db = _conn(q, sql, gold)
        cur = db.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return True, json.dumps(rows, ensure_ascii=False)
    except Exception as e:
        return False, str(e)
    finally:
        try:
            if db:
                db.close()
        except:
            pass


class CoreAgent:
    # simple independent agent: planner output -> tool -> final
    def __init__(self, task: str):
        self.task = task
        self.serper_key = os.environ.get("SERPER_API_KEY", "").strip()

    def _calc(self, text: str) -> str:
        expr = re.sub(r"[^0-9+\-*/().\s]", "", text).strip()
        if not expr:
            return ""
        try:
            return str(eval(expr, {"__builtins__": {}}, {}))
        except:
            return ""

    def _sql(self, q: str, p: str) -> str:
        a = _ans(p)
        if a and "select" in a.lower():
            return a
        t = f"{q}\n{p}".lower()
        if "count" in t:
            return "SELECT COUNT(*) FROM table_name"
        if "avg" in t or "average" in t:
            return "SELECT AVG(column_name) FROM table_name"
        return "SELECT * FROM table_name"

    def _search_query(self, question: str, planner_out: str) -> str:
        txt = str(planner_out)
        for pat in [r"<search>(.*?)</search>", r"search\s*:\s*(.*)", r"query\s*:\s*(.*)"]:
            m = re.search(pat, txt, re.IGNORECASE | re.DOTALL)
            if m:
                q = m.group(1).strip()
                if q:
                    return q
        return question.strip()

    def _wiki_search(self, query: str) -> str:
        try:
            q = urllib.parse.quote(query)
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{q}"
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return str(data.get("extract", ""))[: CONFIGS["SEARCH_LIMIT"]]
        except:
            return ""

    def _serper_search(self, query: str) -> str:
        if not self.serper_key:
            return ""
        try:
            payload = json.dumps({"q": query, "num": 5}).encode("utf-8")
            req = urllib.request.Request(
                "https://google.serper.dev/search",
                data=payload,
                headers={
                    "X-API-KEY": self.serper_key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            parts = []
            for x in data.get("organic", [])[:3]:
                parts.append(f"{x.get('title', '')} {x.get('snippet', '')}")
            return " ".join(parts)[: CONFIGS["SEARCH_LIMIT"]]
        except:
            return ""

    def _search(self, question: str, planner_out: str) -> tuple[str, str]:
        q = self._search_query(question, planner_out)
        txt = self._serper_search(q)
        if txt:
            return q, txt
        return q, self._wiki_search(q)

    def run(self, question: str, planner_out: Any) -> dict[str, Any]:
        a = _ans(planner_out)
        if self.task == "step5":
            need_search = bool(re.search(r"<search>.*?</search>", str(planner_out), re.IGNORECASE | re.DOTALL))
            if a and a != str(planner_out).strip() and not need_search:
                return {
                    "used_tool": False,
                    "tool_name": "",
                    "tool_input": "",
                    "tool_output": "",
                    "final_answer": a,
                }
            q, s = self._search(question, str(planner_out))
            c = self._calc(str(planner_out))
            final = c if c else (a if a else s[: CONFIGS["TOOL_TEXT"]].strip())
            return {
                "used_tool": True,
                "tool_name": "serper_search" if self.serper_key else "wiki_search",
                "tool_input": q,
                "tool_output": s,
                "final_answer": final,
            }
        final = self._sql(question, str(planner_out))
        ok, res = _sql_run(question, final, final)
        return {
            "used_tool": True,
            "tool_name": "sql_executor",
            "tool_input": question,
            "tool_output": res,
            "tool_ok": ok,
            "final_answer": final,
        }


def reward_step5(completions, ground_truth, question, **kwargs):
    ag = CoreAgent("step5")
    out = []
    for c, gt, q in zip(completions, ground_truth, question):
        rs = ag.run(str(q), c)
        pred = _norm(rs["final_answer"])
        gold = _norm(gt)
        score = 0.0
        if pred == gold:
            score += 1.0
        elif gold and gold in _norm(rs["tool_output"]):
            score += 0.6
        if rs["used_tool"] and rs["tool_output"]:
            score += 0.1
        out.append(min(score, 1.0))
    return out


def reward_step6(completions, ground_truth, question, **kwargs):
    ag = CoreAgent("step6")
    out = []
    for c, gt, q in zip(completions, ground_truth, question):
        rs = ag.run(str(q), c)
        p = _norm(rs["final_answer"])
        g = _norm(gt)
        if "select" not in g:
            out.append(1.0 if p == g else 0.0)
            continue
        if p == g:
            out.append(1.0)
            continue
        ok_p, res_p = _sql_run(str(q), rs["final_answer"], str(gt))
        ok_g, res_g = _sql_run(str(q), str(gt), str(gt))
        if ok_p and ok_g and _norm(res_p) == _norm(res_g):
            out.append(0.8)
            continue
        elif ("select" in p and "from" in p) and any(tok in p for tok in g.split()[:3]):
            out.append(0.4)
        else:
            out.append(0.0)
    return out


def test_parse():
    ag = CoreAgent("step6")
    # print(2)
    x = ag.run("Count all users in table users", "<answer>SELECT COUNT(*) FROM users</answer>")
    print(x)


if __name__ == "__main__":
    test_parse()

