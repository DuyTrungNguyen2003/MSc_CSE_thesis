import argparse

from train import run_baseline_training


def str2bool(v):
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False
    else:
        raise argparse.ArgumentTypeError('Enter True or False as argument')


def parse_args():
    parser = argparse.ArgumentParser(description='L0 Gating Baseline Training')
    parser.add_argument('--seed',          type=int,       default=[11, 222, 333, 4444, 55555], nargs='+')
    parser.add_argument('--k',             type=int,       default=25587)
    parser.add_argument('--lamba',         type=int,     default=[200], nargs='+')
    parser.add_argument('--droprate_init', type=float,     default=0.5)
    parser.add_argument('--learning_rate', type=float,     default=1e-2)
    parser.add_argument('--use_wandb',     type=str2bool,  default=False)
    parser.add_argument('--extract_features', type=str2bool,  default=False)
    return parser.parse_args()


def print_seed_summary(seed, lamba, results, k):
    fold_results  = results['fold_results']
    avg_auroc     = results['avg_test_auroc']
    avg_selected  = k - results['avg_pruned_features']

    print(f"\n{'#'*60}")
    print(f"  SUMMARY  |  seed={seed}  lamba={lamba}")
    print(f"{'#'*60}")
    print(f"  {'Fold':<8} {'Test AUROC':>12} {'Selected Features':>20}")
    print(f"  {'-'*42}")
    for fold_idx, test_auroc, pruned in fold_results:
        print(f"  {fold_idx:<8} {test_auroc:>12.4f} {k - pruned:>20d}")
    print(f"  {'-'*42}")
    print(f"  {'Average':<8} {avg_auroc:>12.4f} {avg_selected:>20.1f}")
    print(f"{'#'*60}\n")


def print_all_runs_summary(all_results, k):
    print(f"\n{'#'*60}")
    print(f"  FINAL SUMMARY  |  All Runs")
    print(f"{'#'*60}")

    for seed, lamba, results in all_results:
        fold_results  = results['fold_results']
        avg_auroc     = results['avg_test_auroc']
        avg_selected  = k - results['avg_pruned_features']

        print(f"\n  seed={seed}  lamba={lamba}")
        print(f"  {'Fold':<8} {'Test AUROC':>12} {'Selected Features':>20}")
        print(f"  {'-'*42}")
        for fold_idx, test_auroc, pruned in fold_results:
            print(f"  {fold_idx:<8} {test_auroc:>12.4f} {k - pruned:>20d}")
        print(f"  {'-'*42}")
        print(f"  {'Average':<8} {avg_auroc:>12.4f} {avg_selected:>20.1f}")

    print(f"\n{'#'*60}\n")


if __name__ == "__main__":
    args = parse_args()
    all_results = []
    for seed in args.seed:
        for lamba in args.lamba:
            print(f"\nRunning CV training — seed={seed}  lamba={lamba}")
            results = run_baseline_training(
                seed=seed,
                L0_LAMBDA=lamba,
                droprate_init=args.droprate_init,
                learning_rate=args.learning_rate,
                k=args.k,
                use_wandb=args.use_wandb,
                extract_features=args.extract_features,
            )
            print_seed_summary(seed, lamba, results, args.k)
            all_results.append((seed, lamba, results))

    print_all_runs_summary(all_results, args.k)