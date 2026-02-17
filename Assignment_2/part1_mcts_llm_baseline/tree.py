import math


class QNode:
    # question node
    def __init__(self, question, possibilities, yes_items, no_items, parent=None):
        self.question = question
        self.parent = parent
        self.possibilities = possibilities
        self.yes_items = yes_items
        self.no_items = no_items

        self.yes_child = RNode(yes_items, response="YES", parent=self)
        self.no_child = RNode(no_items, response="NO", parent=self)

        self.r_ig = info_gain(len(possibilities), len(yes_items), len(no_items))
        self.r_accumulated = 0.0
        self.r_expected = 0.0

    def __repr__(self):
        return f"Q[{self.question[:40]}] Y:{len(self.yes_items)} N:{len(self.no_items)}"


class RNode:
    # answer node
    def __init__(self, possibilities, response, parent=None):
        self.possibilities = possibilities
        self.response = response
        self.parent = parent
        self.children = []

    def is_terminal(self):
        return len(self.possibilities) <= 2

    def __repr__(self):
        return f"R[{self.response}] |O|={len(self.possibilities)}"


def info_gain(total, n_yes, n_no, lam=0.4):
    # information gain R_IG -- formula(3)
    if total == 0 or n_yes == 0 or n_no == 0:
        return 0.0

    p_a = n_yes / total
    p_n = n_no / total

    ent = -p_a * math.log2(p_a) - p_n * math.log2(p_n)
    # 0.4 is lambda in the paper
    penalty = 1 + (1 / lam) * abs(p_a - p_n)
    return ent / penalty


def acc_reward(node):
    # accumulated reward -- formula(4)
    if node.parent is None:
        return node.r_ig

    r_node = node.parent
    q_node = r_node.parent
    if q_node is None:
        return node.r_ig
    return node.r_ig + q_node.r_accumulated


def exp_reward(q_node):
    # expected reward -- formula(5)(6)
    total = len(q_node.possibilities)
    if total == 0:
        return 0.0

    p_a = len(q_node.yes_items) / total
    p_n = len(q_node.no_items) / total

    yc = q_node.yes_child
    nc = q_node.no_child

    if yc.is_terminal() or len(yc.children) == 0:
        r_yes = q_node.r_accumulated
    else:
        r_yes = sum(c.r_expected for c in yc.children) / len(yc.children)

    if nc.is_terminal() or len(nc.children) == 0:
        r_no = q_node.r_accumulated
    else:
        r_no = sum(c.r_expected for c in nc.children) / len(nc.children)

    return p_a * r_yes + p_n * r_no


def get_context(node):
    # get the context of the ancestors and put it in the prompt
    parts = []
    cur = node
    while cur is not None:
        if isinstance(cur, RNode) and cur.parent is not None:
            q = cur.parent.question
            a = cur.response
            parts.append(f"{q} {a}")
            cur = cur.parent.parent
        elif isinstance(cur, QNode) and cur.parent is not None:
            cur = cur.parent
        else:
            break

    parts.reverse()
    if parts:
        return "For context, following questions were already asked to build the above set of possibilities: " + "; ".join(parts)
    return ""


if __name__ == "__main__":
    print(f"2+2: IG = {info_gain(4, 2, 2):.4f}")
    print(f"3+1: IG = {info_gain(4, 3, 1):.4f}")
    print(f"4+0: IG = {info_gain(4, 4, 0):.4f}")

    omega = ["flu", "pneumonia", "enteritis", "asthma", "gastritis"]
    q = QNode("Do you have difficulty breathing?", omega,
              ["pneumonia", "asthma"], ["flu", "enteritis", "gastritis"])
    print(f"\n{q}")
    print(f"YES: {q.yes_child}, terminal={q.yes_child.is_terminal()}")

    # print(info_gain(10, 5, 5))
    # print(info_gain(10, 9, 1))
    # print(info_gain(6, 3, 3))
