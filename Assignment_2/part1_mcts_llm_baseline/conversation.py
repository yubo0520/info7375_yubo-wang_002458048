from tree import RNode
from uot import expand_tree, pick_best, make_guess


def answerer_prompt(target, domain="20q"):
    if domain == "md":
        return f"""You are a patient suffering from '{target}', and communicating with a doctor.
I will ask you questions, and you should answer each one truthfully based on your disease, by saying Yes or No.
Note that you must never reveal the disease until the doctor correctly mentions or asks about it (e.g. "Are you a [disease name]?").
If the doctor correctly guesses your disease in their question, you must directly respond "You are right. I am experiencing '{target}'."
"""
    elif domain == "ts":
        return f"""You are a client with a device that has '{target}' and I am the technician.
I will ask you questions, and you should answer each one truthfully based on the issue of your device, by saying Yes or No.
Note that you must never reveal the issue name until I tell it correctly.
If I tell your issue correctly in my question, directly respond: "You are right. My device has issues with '{target}'."
"""
    else:
        return f"""Let us play the game of 20 questions. You are the answerer and I am questioner.
X is '{target}'. I will ask you questions and you should answer each one truthfully based on being X, by saying Yes or No.
Note that you must never reveal X, until I guess it correctly.
If I guess X correctly in my question, directly respond "You guessed it. X is '{target}'."
"""


def get_answer(ans_llm, history, question):
    history.append({"role": "user", "content": question})
    resp = ans_llm.chat(history)
    history.append({"role": "assistant", "content": resp})

    lo = resp.lower()
    hit = ("you are right" in lo or "you guessed it" in lo)
    return resp, hit


def parse_yn(text):
    # parse YES/NO from answer
    t = text.strip().lower()
    if "you are right" in t or "you guessed it" in t:
        return "YES"
    if t.startswith("yes"):
        return "YES"
    if t.startswith("no"):
        return "NO"
    # sometimes model answers are verbose, can only check if it contains
    if "yes" in t and "no" not in t:
        return "YES"
    if "no" in t and "yes" not in t:
        return "NO"
    return "NO"  # default NO, if unsure, it's better to be wrong than right


# --- UoT 

def run_uot(q_llm, a_llm, sample, omega, domain, config):
    sid, idx, self_repo, target = sample

    T = config.get("max_turns", 6)
    ds = config.get("tree_depth", 3)
    m = config.get("n_questions", 3)

    a_hist = [{"role": "system", "content": answerer_prompt(target, domain)}]
    cur = RNode(possibilities=list(omega), response="ROOT")

    conv = []
    qgc = {"count": 0}
    success = False
    info_turns = int(config.get("delta", 0.6) * T)  # first 60% turns for information gathering

    print(f"\n--- Sample #{idx}: target='{target}' ---")
    print(f"  self_repo: {self_repo[:80]}...")
    print(f"  |O| = {len(omega)}, T = {T}, info_turns = {info_turns}")

    best_q = None
    guessed = None
    used_tree = False

    for turn in range(T):
        print(f"\n  Turn {turn+1}/{T}, |O_cur| = {len(cur.possibilities)}")

        if turn < info_turns and not cur.is_terminal():
            expand_tree(q_llm, cur, depth=0, max_depth=ds, m=m,
                        domain=domain, qgc=qgc)
            best_q = pick_best(cur)

            if best_q is None:
                print("    [no question, targeting]")
                question, guessed = make_guess(q_llm, cur.possibilities, domain)
                used_tree = False
            else:
                question = best_q.question
                guessed = None
                used_tree = True
                print(f"    Q: {question[:80]}...")
                print(f"    R_exp: {best_q.r_expected:.4f}")
        else:
            question, guessed = make_guess(q_llm, cur.possibilities, domain)
            used_tree = False
            if question is None:
                print(f"    [Targeting] no candidates left")
                break
            print(f"    [Targeting] Q: {question}")

        resp, hit = get_answer(a_llm, a_hist, question)
        print(f"    A: {resp[:100]}")

        conv.append({
            "turn": turn + 1,
            "question": question,
            "response": resp,
            "phase": "info_seeking" if turn < info_turns else "targeting"
        })

        if hit:
            success = True
            print(f"    >> Success at turn {turn+1}")
            break

        if used_tree and best_q is not None:
            yn = parse_yn(resp)
            if yn == "YES":
                cur = best_q.yes_child
            else:
                cur = best_q.no_child
            print(f"    -> {yn}, |O| = {len(cur.possibilities)}")
            print(f"    -> left: {cur.possibilities[:5]}{'...' if len(cur.possibilities) > 5 else ''}")
        elif guessed is not None:
            left = [p for p in cur.possibilities if p != guessed]
            cur = RNode(possibilities=left, response="UPDATED")
            print(f"    -> wrong guess, |O| = {len(left)}")

    if not success:
        print(f"    >> Failed after {T} turns")

    return {
        "success": 1 if success else 0,
        "num_turns": len(conv),
        "qgc": qgc["count"],
        "conversation": conv
    }


#  DP 

def run_dp(q_llm, a_llm, sample, omega, domain, config):
    sid, idx, self_repo, target = sample
    T = config.get('max_turns', 6)

    if domain == "md":
        diseases = ", ".join(omega)
        sys_p = f"""You are a doctor and your patient reports that: {self_repo}
You should ask your patient questions in English with symptoms which can only be answered by yes or no, to find what disease this patient suffers.
Based on the symptoms above, if you find out the disease, please ask 'Are you a [disease name]?'
The patient may suffer from one of the diseases below: {diseases}"""
    elif domain == "ts":
        sys_p = f"""You are a technician and your client reports that: {self_repo}
You should ask questions which can only be answered by yes or no, to find what issue the client faces.
The client may face one of these issues: {', '.join(omega)}."""
    else:
        sys_p = f"""Let us play 20 questions. I am impersonating the thing X.
X is possibly one of: {', '.join(omega)}.
Ask yes/no questions starting with 'Is X'. Ask one question at a time."""

    q_hist = [{"role": "system", "content": sys_p}]
    a_hist = [{"role": "system", "content": answerer_prompt(target, domain)}]

    conv = []
    success = False
    print(f"\n--- [DP] Sample #{idx}: target='{target}' ---")

    for turn in range(T):
        if turn == 0:
            q_hist.append({"role": "user", "content": "Let us begin. Ask me the first question."})

        question = q_llm.chat(q_hist)
        q_hist.append({"role": "assistant", "content": question})

        print(f"  Turn {turn+1}: Q: {question[:100]}")

        resp, hit = get_answer(a_llm, a_hist, question)
        print(f"           A: {resp[:100]}")

        # start reminding from turn>=2 (phase 2)
        if turn >= 2:
            diseases_str = ", ".join(omega)
            reminder = f"Note that you should try to confirm the disease soon. The patient may suffer from one of the diseases below: {diseases_str}. Ask your next question."
            q_hist.append({"role": "user", "content": f"{resp}\n\n{reminder}"})
        else:
            q_hist.append({"role": "user", "content": resp})

        conv.append({
            "turn": turn + 1,
            "question": question,
            "response": resp
        })

        if hit:
            success = True
            print(f"    >> Success!")
            break

    return {
        "success": 1 if success else 0,
        "num_turns": len(conv),
        "qgc": 0,
        "conversation": conv
    }
