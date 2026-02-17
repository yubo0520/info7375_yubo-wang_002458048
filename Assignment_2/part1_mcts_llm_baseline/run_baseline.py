import argparse
from llm import LLM
from db import load_samples, load_omega, save, calc_metrics
from conversation import run_uot, run_dp

ALL_MODELS = ["llama3.1:8b", "qwen2.5:7b"]
CONFIGS = {
    "DX": {"domain": "md", "max_turns": 6, "delta": 0.6, "tree_depth": 3, "n_questions": 3},
}


def run_experiment(method, dataset, model, samples, omega, cfg):
    # run a combination of model+method+dataset
    domain = cfg["domain"]

    print(f"\n{'#'*60}")
    print(f"# model={model}, method={method}, dataset={dataset}, n={len(samples)}")
    print(f"# omega={omega}")
    print(f"{'#'*60}")

    q_llm = LLM(model)
    a_llm = LLM(model)

    for i, sample in enumerate(samples):
        print(f"\n{'='*50}")
        print(f"Progress: {i+1}/{len(samples)} [{model}]")

        if method == "DP":
            res = run_dp(q_llm, a_llm, sample, omega, domain, cfg)
        else:
            res = run_uot(q_llm, a_llm, sample, omega, domain, cfg)

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
    parser.add_argument("--method", type=str, choices=["dp", "uot"], required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default=None,
                        help="single model, or 'all' to run ALL_MODELS")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tree_depth", type=int, default=None)
    parser.add_argument("--n_questions", type=int, default=None)
    args = parser.parse_args()

    if args.dataset not in CONFIGS:
        print(f"Unknown dataset: {args.dataset}, available: {list(CONFIGS.keys())}")
        return

    cfg = CONFIGS[args.dataset].copy()
    if args.tree_depth is not None:
        cfg["tree_depth"] = args.tree_depth
    if args.n_questions is not None:
        cfg["n_questions"] = args.n_questions

    method = args.method.upper()

    samples = load_samples(args.dataset, limit=args.limit)
    omega = load_omega(args.dataset)

    if not samples:
        print(f"No samples in '{args.dataset}'. Run setup_db.py first.")
        return
    if not omega:
        print(f"Empty omega for '{args.dataset}'.")
        return

    # decide which models to run
    if args.model is None or args.model == "all":
        models = ALL_MODELS
        print(f"Running all models: {models}")
    else:
        models = [args.model]

    for model in models:
        run_experiment(method, args.dataset, model, samples, omega, cfg)

    # a total comparison
    if len(models) > 1:
        print(f"\n\n{'='*60}")
        print(f"  ALL MODELS COMPARISON: {args.dataset} / {method}")
        print(f"{'='*60}")
        for model in models:
            calc_metrics(args.dataset, method, model)


if __name__ == "__main__":
    main()
