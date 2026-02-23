import math
import random
from tree import QNode, RNode, acc_reward, exp_reward, get_context, node_depth
from uot import gen_questions

def uct(node, parent_visits, c=0.2, cluster_id=None):
    # formula(7) + (13)
    if node.visits == 0:
        return float('inf')
    exploit = node.total_reward / node.visits
    explore = c * math.sqrt(math.log(parent_visits) / node.visits)
    bonus = 0
    if cluster_id is not None:
        bonus = node.cluster_bonus.get(cluster_id, 0)
    return exploit + explore + bonus

def selection(root, c=0.2, cluster_id=None):
    cur = root
    while True:
        if isinstance(cur, RNode):
            if not cur.children or cur.is_terminal():
                return cur
            pv = max(cur.visits, 1)  # avoid log(0)
            best = max(cur.children,
                       key=lambda q: uct(q, pv, c=c, cluster_id=cluster_id))
            cur = best
        else:
            if cur.yes_child.visits == 0 and cur.no_child.visits == 0:
                cur = random.choice([cur.yes_child, cur.no_child])
            elif cur.yes_child.visits == 0:
                cur = cur.yes_child
            elif cur.no_child.visits == 0:
                cur = cur.no_child
            else:
                pv = max(cur.visits, 1)
                y_score = uct(cur.yes_child, pv, c=c, cluster_id=cluster_id)
                n_score = uct(cur.no_child, pv, c=c, cluster_id=cluster_id)
                if y_score >= n_score:
                    cur = cur.yes_child
                else:
                    cur = cur.no_child

def expansion(llm, rnode, m, domain, qgc):
    # one-level expand only (lazy)
    if rnode.is_terminal() or rnode.children:
        return
    ctx = get_context(rnode)
    candidates = gen_questions(llm, rnode.possibilities, ctx, m=m, domain=domain)
    qgc["count"] += 1
    if not candidates:
        print(f"    [expansion] no candidates for |O|={len(rnode.possibilities)}")
        return
    for q_text, yi, ni in candidates:
        qn = QNode(q_text, rnode.possibilities, yi, ni, parent=rnode)
        qn.r_accumulated = acc_reward(qn)
        rnode.children.append(qn)
    for qn in rnode.children:
        qn.r_expected = exp_reward(qn)


def simulation(node, ds=3):
    cur = node
    depth = 0
    while depth < ds:
        if isinstance(cur, RNode):
            if cur.is_terminal() or not cur.children:
                break
            cur = random.choice(cur.children)
        else:
            cur = random.choice([cur.yes_child, cur.no_child])
        depth += 1

    if isinstance(cur, QNode):
        return cur.r_expected
    if cur.children:
        return sum(q.r_expected for q in cur.children) / len(cur.children)
    if cur.parent and isinstance(cur.parent, QNode):
        return cur.parent.r_accumulated
    return 0.0

def backpropagation(node, reward):
    # formula(8)
    cur = node
    while cur is not None:
        cur.visits += 1
        cur.total_reward += reward
        cur = cur.parent



def propagate_feedback(leaf_node, cluster_id, target, beta=0.2, gamma=0.9):
    # formula(12), only nodes containing target
    cur = leaf_node
    while cur is not None:
        d = node_depth(cur)
        bonus = beta * max(cur.total_reward, 0) * (gamma ** d)
        if target in cur.possibilities:
            if cluster_id not in cur.cluster_bonus:
                cur.cluster_bonus[cluster_id] = 0.0
            cur.cluster_bonus[cluster_id] += bonus
        cur = cur.parent


# --- MISQ search (no feedback)

def misq_search(llm, root, m, domain, qgc, n_iter=10, c=0.2, cluster_id=None):
    if root.is_terminal():
        return None

    if not root.children:
        expansion(llm, root, m, domain, qgc)
    if not root.children:
        return None

    for _ in range(n_iter):
        leaf = selection(root, c=c, cluster_id=cluster_id)
        if not leaf.is_terminal() and not leaf.children:
            expansion(llm, leaf, m, domain, qgc)
        r = simulation(leaf)
        backpropagation(leaf, r)

    # total_reward works better than r_expected here
    if cluster_id is not None:
        best = max(root.children,
                   key=lambda q: q.total_reward + q.cluster_bonus.get(cluster_id, 0))
    else:
        best = max(root.children, key=lambda q: q.total_reward)
    # print(f"  misq: best='{best.question[:40]}' v={best.visits} r={best.total_reward:.3f}")
    return best


if __name__ == "__main__":
    from tree import RNode, QNode
    omega = ["flu", "pneumonia", "enteritis", "asthma", "gastritis"]
    root = RNode(omega, "ROOT")

    q = QNode("Do you have fever?", omega,
              ["flu", "pneumonia"], ["enteritis", "asthma", "gastritis"],
              parent=root)
    root.children.append(q)

    # test feedback propagation
    q.visits = 5
    q.total_reward = 2.5
    propagate_feedback(q.yes_child, cluster_id=0, target="flu", beta=0.2, gamma=0.9)
    print(f"q bonus after feedback: {q.cluster_bonus}")
    print(f"root bonus: {root.cluster_bonus}")

    # test UCT with bonus
    print(f"UCT w/o cluster: {uct(q, 10, c=0.2):.4f}")
    print(f"UCT w/ cluster 0: {uct(q, 10, c=0.2, cluster_id=0):.4f}")
    print(f"UCT w/ cluster 99: {uct(q, 10, c=0.2, cluster_id=99):.4f}")
