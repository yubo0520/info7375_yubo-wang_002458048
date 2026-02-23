import argparse
from llm import LLM
from db import load_samples, load_omega, save, calc_metrics
from tree import RNode
from conversation import run_dp, run_uot, run_misq, run_misq_hf

ALL_MODELS = ["llama3.1:8b", "qwen2.5:7b"]

# only DX for now
CONFIGS = {
    "DX": {
        "domain": "md", "max_turns": 6, "delta": 0.6,
        "tree_depth": 3, "n_questions": 3,
        # mcts params
        "n_iter": 10, "mcts_c": 0.2,
        # feedback params
        "beta": 0.2, "gamma": 0.9, "tau": 0.9,
    },
    # "MedDG": {
    #     "domain": "md", "max_turns": 6, "delta": 0.6,
    #     "tree_depth": 3, "n_questions": 3,
    #     "n_iter": 10, "mcts_c": 0.2,
    #     "beta": 0.2, "gamma": 0.9, "tau": 0.9,
    # },
}

def run_experiment(method, dataset, model, samples, omega, cfg):
    domain = cfg["domain"]

    print(f"\n{'#'*60}")
    print(f"# model={model}, method={method}, dataset={dataset}, n={len(samples)}")
    print(f"{'#'*60}")

    q_llm = LLM(model)
    a_llm = LLM(model)

    shared_root = None
    embedder = None
    clusters = {}

    if method in ("MISQ", "MISQ-HF"):
        shared_root = RNode(possibilities=list(omega), response="ROOT")
        print(f"  shared tree root: |O| = {len(omega)}")

    if method == "MISQ-HF":
        from embedding import Embedder
        embedder = Embedder()
        print("  embedding model loaded")

    for i, sample in enumerate(samples):
        print(f"\n{'='*50}")
        print(f"Progress: {i+1}/{len(samples)} [{model}]")

        if method == "DP":
            res = run_dp(q_llm, a_llm, sample, omega, domain, cfg)
        elif method == "UOT":
            res = run_uot(q_llm, a_llm, sample, omega, domain, cfg)
        elif method == "MISQ":
            res = run_misq(q_llm, a_llm, sample, omega, domain, cfg,
                           shared_root)
        elif method == "MISQ-HF":
            res = run_misq_hf(q_llm, a_llm, sample, omega, domain, cfg,
                              shared_root, embedder, clusters)
        else:
            print(f"unknown method: {method}")
            return

        save(
            sample_id=sample[0], ds_name=dataset,
            method=method, model=model, omega_aware=1,
            success=res["success"], n_turns=res["num_turns"],
            qgc=res["qgc"], conv_log=res["conversation"]
        )

    print(f"\n--- {model} done ---")
    calc_metrics(dataset, method, model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str,
                        choices=["dp", "uot", "misq", "misq-hf"],
                        required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n_iter", type=int, default=None,
                        help="MCTS iterations (default 10)")
    parser.add_argument("--mcts_c", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None,
                        help="feedback scaling factor")
    parser.add_argument("--gamma", type=float, default=None,
                        help="feedback depth decay")
    parser.add_argument("--tau", type=float, default=None,
                        help="cluster similarity threshold")
    args = parser.parse_args()

    if args.dataset not in CONFIGS:
        print(f"Unknown dataset: {args.dataset}")
        return

    cfg = CONFIGS[args.dataset].copy()
    for k in ["n_iter", "mcts_c", "beta", "gamma", "tau"]:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    method_map = {
        "dp": "DP",
        "uot": "UOT",
        "misq": "MISQ",
        "misq-hf": "MISQ-HF",
    }
    method = method_map[args.method]

    samples = load_samples(args.dataset, limit=args.limit)
    omega = load_omega(args.dataset)

    if not samples:
        print(f"No samples. Run setup_db.py first.")
        return

    if args.model is None or args.model == "all":
        models = ALL_MODELS
    else:
        models = [args.model]

    for model in models:
        run_experiment(method, args.dataset, model, samples, omega, cfg)


if __name__ == "__main__":
    main()
