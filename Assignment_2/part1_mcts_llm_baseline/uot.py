import re
import time
from tree import QNode, RNode, acc_reward, exp_reward, get_context


def _match_item(item, possibilities):
    # fuzzy matching
    item = item.strip()
    if item in possibilities:
        return item
    lo = item.lower().replace(",", "").replace(" and ", " ")
    for p in possibilities:
        pl = p.lower().replace(",", "").replace(" and ", " ")
        if lo == pl or lo in pl or pl in lo:
            return p
    return None


def _try_simple_parse(text, possibilities):
    questions = []
    chunks = text.split("Question")
    for chunk in chunks:
        if "YES:" not in chunk or "NO:" not in chunk:
            continue
        # get question
        lines = chunk.strip().split("\n")
        q_text = ""
        for l in lines:
            if "?" in l:
                q_text = l.split(":", 1)[-1].strip().strip('"').strip("*")
                break
        if not q_text:
            continue
        # get YES items
        yes_part = chunk.split("YES:")[1].split("\n")[0]
        no_part = chunk.split("NO:")[1].split("\n")[0]
        yi = []
        ni = []
        for it in yes_part.split(","):
            m = _match_item(it.strip(), possibilities)
            if m: yi.append(m)
        for it in no_part.split(","):
            m = _match_item(it.strip(), possibilities)
            if m: ni.append(m)
        # remove duplicates
        yi = list(dict.fromkeys(yi))
        ni = list(dict.fromkeys(ni))
        # fill in missing items
        all_found = set(yi + ni)
        missing = [p for p in possibilities if p not in all_found]
        if missing:
            if len(yi) <= len(ni):
                yi.extend(missing)
            else:
                ni.extend(missing)
        if q_text and yi and ni:
            questions.append((q_text, yi, ni))
    return questions


def parse_response(text, possibilities):
    # parse the response from LLM that generates questions

    try:
        simple = _try_simple_parse(text, possibilities)
        if simple:
            # print("simple parse ok")
            return simple
    except:
        pass  # simple method failed, use the following method

    # print(2)  # simple method failed, use the following method
    questions = []

    blocks = re.split(r'Question\s*\d+\s*:', text, flags=re.IGNORECASE)
    if len(blocks) <= 1:
        blocks = re.split(r'Q\s*\d+\s*:', text, flags=re.IGNORECASE)
    if len(blocks) <= 1:
        blocks = re.split(r'\n\s*\d+\.\s+', text)

    for block in blocks:
        if not block.strip():
            continue

        lines = block.strip().split('\n')
        q_text = ""
        yes_items = []
        no_items = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if '?' in line and not q_text:
                q = line.strip()
                for pat in [r'["\']([^"\']*\?)["\']', r'\*+([^*]*\?)\*+', r'[:\*]\s*["\']?([^"\'*\n]*\?)["\']?']:
                    m = re.search(pat, q)
                    if m:
                        q = m.group(1).strip()
                        break
                q = re.sub(r'^[\d\.\s]+', '', q).strip()
                if q and len(q) > 5:
                    q_text = q

            elif re.match(r'^YES\s*:', line, re.IGNORECASE):
                items_str = line.split(':', 1)[1].strip()
                for it in re.split(r'[,;]', items_str):
                    n = _match_item(it, possibilities)
                    if n:
                        yes_items.append(n)

            elif re.match(r'^NO\s*:', line, re.IGNORECASE):
                items_str = line.split(':', 1)[1].strip()
                for it in re.split(r'[,;]', items_str):
                    n = _match_item(it, possibilities)
                    if n:
                        no_items.append(n)

        yes_items = list(dict.fromkeys(yes_items))
        no_items = list(dict.fromkeys(no_items))

        if q_text and (yes_items or no_items):
            vy = [x for x in yes_items if x in possibilities]
            vn = [x for x in no_items if x in possibilities]

            assigned = set(vy + vn)
            missing = [p for p in possibilities if p not in assigned]
            if missing:
                if len(vy) <= len(vn):
                    vy.extend(missing)
                else:
                    vn.extend(missing)

            if vy and vn:
                questions.append((q_text, vy, vn))

    # last fallback: pick a question at random
    if not questions and possibilities and '?' in text:
        m = re.search(r'["\*]([^"*\n]{10,}?\?)["\*]', text)
        if not m:
            m = re.search(r'([^.!?\n]{15,}?\?)', text)
        if m:
            qt = m.group(1).strip()
            if len(qt) > 5:
                mid = (len(possibilities) + 1) // 2
                questions.append((qt, possibilities[:mid], possibilities[mid:]))

    return questions


def gen_questions(llm, possibilities, context, m=3, domain="20q"):
    # generate m candidate questions using LLM
    items_str = ", ".join(possibilities)

    if domain == "md":
        prompt = f"""You are a doctor. Here are all the possible diseases that the patient may suffer from: {items_str}
Design a question to ask your patient regarding symptoms of their illness that can only be answered by Yes or No. Then classify the possible diseases above based on this question. If the answer is 'YES', put this disease into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many diseases are in YES and NO. Notably, this question should fulfill that the count of YES and NO are almost the same with a permissible discrepancy of no more than one!
{context}
Based on this information, create most relevant {m} questions to ask (and classify the above diseases). Your response should strictly follow the template:

Question 1: ...?
YES: comma-separated, list of disease names, ...
Count of YES: ...
NO: comma-separated, list of disease names, ...
Count of NO: ..."""
    elif domain == "ts":
        prompt = f"""You are a technician. Here are all the issues that the client may face: {items_str}
Design a question to ask your client with a specific situation that can only be answered by YES or NO. Then classify the possible issues above based on this question. If the answer is 'YES', put this issue into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many issues are in YES and NO. Notably, this question should fulfill that the count of YES and NO are almost the same with a permissible discrepancy of no more than one!
{context}
Based on this information, create the most relevant {m} questions to classify the above issues correctly. Your response should strictly follow the template:

Question 1: ...?
YES: comma-separated, list of issue names, ...
Count of YES: ...
NO: comma-separated, list of issue names, ...
Count of NO: ..."""
    else:
        prompt = f"""Here are all the X: {items_str}
Design a question about X that can only be answered by Yes or No. Then classify the possible X above based on this question. If the answer is 'YES', put this X into 'YES: ...', otherwise to 'NO: ...'. Finally calculate how many X in YES and NO. Notably, this question should fulfill that the count of YES and NO are almost the same with a permissible discrepancy of no more than one!
{context}
Based on this information, create most relevant {m} questions to classify the above X correctly. Your response should strictly follow the template:

Question 1: Is X ...?
YES: comma-separated, list of things, ...
Count of YES: ...
NO: comma-separated, list of things, ...
Count of NO: ..."""

    try:
        resp = llm.chat([{"role": "user", "content": prompt}])
        # print([resp])
        parsed = parse_response(resp, possibilities)
        return parsed[:m]
    except:
        # if error, retry, usually won't reach here
        print("gen_questions failed, retrying")
        time.sleep(1)
        return gen_questions(llm, possibilities, context, m, domain)


#  tree expansion 

def expand_tree(llm, rnode, depth, max_depth, m, domain, qgc):
    # exhaustive expansion of decision tree
    if depth >= max_depth:
        return
    if rnode.is_terminal():
        return
    if len(rnode.children) > 0:
        return

    ctx = get_context(rnode)

    candidates = gen_questions(llm, rnode.possibilities, ctx, m=m, domain=domain)
    qgc["count"] += 1

    if not candidates:
        return

    for q_text, yi, ni in candidates:
        qn = QNode(
            question=q_text, possibilities=rnode.possibilities,
            yes_items=yi, no_items=ni, parent=rnode
        )
        qn.r_accumulated = acc_reward(qn)
        rnode.children.append(qn)

        expand_tree(llm, qn.yes_child, depth + 1, max_depth, m, domain, qgc)
        expand_tree(llm, qn.no_child, depth + 1, max_depth, m, domain, qgc)

    for qn in rnode.children:
        qn.r_expected = exp_reward(qn)


def pick_best(rnode):
    if not rnode.children:
        return None
    return max(rnode.children, key=lambda q: q.r_expected)


def make_guess(llm, possibilities, domain="20q"):
    # targeting: directly guess the first one
    if not possibilities:
        return None, None
    item = possibilities[0]
    if domain == "md":
        q = f"Are you a {item}?"
    elif domain == "ts":
        q = f"Are you experiencing {item}?"
    else:
        q = f"Is X '{item}'?"
    return q, item


if __name__ == "__main__":
    # test parsing
    sample = """Question 1: Do you have difficulty breathing?
YES: pneumonia, asthma
Count of YES: 2
NO: flu, enteritis, gastritis
Count of NO: 3

Question 2: Do you have fever?
YES: flu, pneumonia
Count of YES: 2
NO: enteritis, asthma, gastritis
Count of NO: 3"""

    omega = ["flu", "pneumonia", "enteritis", "asthma", "gastritis"]
    parsed = parse_response(sample, omega)
    for q, yes, no in parsed:
        print(f"Q: {q}")
        print(f"  YES: {yes}")
        print(f"  NO:  {no}")

# kept for reference
# def test_parse():
#     text = "Question 1: Is it respiratory?\nYES: flu, pneumonia\nNO: enteritis\n"
#     r = parse_response(text, ["flu","pneumonia","enteritis"])
#     print(r)
#     text2 = "1. Do you cough?\nYES: flu\nNO: enteritis"
#     r2 = parse_response(text2, ["flu","enteritis"])
#     print(r2)
# test_parse()
