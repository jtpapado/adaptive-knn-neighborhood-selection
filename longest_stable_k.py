#!/usr/bin/env python3
"""
A window rule with no window: the longest stable run, centred.

The existing window method fixes a length w and takes the end of the FIRST run
of w consecutive correct leave-one-out predictions. Sweeping w over fourteen
datasets showed there is no value that is safe across them — the per-dataset
optima run from 2 to 15 with no pattern, and no single w keeps every dataset
within a point of its own best. w is a parameter that has to be tuned and
cannot be defended as inert.

This rule removes it. For each training point it scans k = 1..k_max, takes the
LONGEST run of consecutive correct predictions, and keeps the middle k of that
run:

    k = 15..25 correct  ->  run of 11  ->  k = 20

Ties go to the smaller k, in both senses: two runs of equal length pick the
earlier one, and an even-length run picks the lower of its two middles. A point
with no correct k at all keeps k_max, as the other rules do.

The reasoning is that the middle of the widest stable band is the k least likely
to sit next to a boundary — where the first-correct rule lands by construction,
and where a fixed window lands whenever w happens to be shorter than the band.

Evaluated against AdaNN (first correct k) and the fixed-window rule on identical
folds, so the three can be compared as paired tests. All three are read off one
correctness matrix, which is what the derive actually costs.

    python3 longest_stable_k.py
    python3 longest_stable_k.py --datasets iris ecl --models CustomKNN
    python3 longest_stable_k.py --figures-only

Needs knn_pipeline.py, knn_algorithms.py and sensitivity_study.py beside it.
"""
import os
import sys
import glob
import time
import json
import argparse
import numpy as np
import pandas as pd

MODELS = ["CustomKNN", "MutualKNN", "SymmetricKNN"]
DATASETS = ["iris", "ecl", "bl", "pm", "ys", "bn", "ph", "txr",
            "ls", "tn", "pd", "mgt", "lir", "sh"]
LABEL = {"CustomKNN": "kNN (directed)", "MutualKNN": "Mutual kNN",
         "SymmetricKNN": "Symmetric kNN"}
COLOUR = {"CustomKNN": "#2f6fdb", "MutualKNN": "#e8703a", "SymmetricKNN": "#1faa72"}
BG = "#fbfaf8"

CONFIG = """# Written by longest_stable_k.py on {stamp}
base_folder: "{base}"
output_folder: "results"
per_dataset_results: false
noise: {{enabled: false, rates: [0.3], seed: 42, run_clean: true}}
datasets:
  include: {datasets}
  exclude: []
models: []
backend: "{backend}"
timing: "{timing}"
gpu_max_elements: 72000000
cv:
  folds: {folds}
  repeats: 1
  bestk_folds: {folds}
  bestk_repeats: 1
  seed: {seed}
  scaling: "minmax"
  rare_class_policy: "warn"
phases: {{phase1: false, phase2: false, phase3: false, phase4: false, phase5: false,
         phase6: false, phase7: false, phase8: false, phase9: false, phase10: false,
         phase11: false, phase12: false, phase13: false}}
phase1: {{k_values: [5]}}
phase2: {{k_min: 1, k_max: 30, cv_folds: 10, tie_breaking: "lowest_k"}}
phase4: {{outer_folds: {folds}, inner_folds: 10, repeats: 1, seed: {seed},
         rare_class_policy: "warn", methods: ["bestk"], w_values: [2,3]}}
phase5: {{k_max_values: [9, 30]}}
phase9: {{k_max: 30, stability_windows: [3]}}
workers: {{main: -1, builders: 1, testers: 10}}
"""


# ===========================================================================
# THE RULE
# ===========================================================================

def longest_stable_k(correct):
    """
    Middle k of the longest run of consecutive correct predictions, per point.

    Returns (k_per_row, run_length, run_start), all 1-indexed in k.

    The scan keeps a running length and only replaces the best on a strict
    improvement, so equal-length runs leave the earliest one standing — the
    smaller k. The middle uses (length - 1) // 2, which on an even run picks
    the lower of the two central values, again the smaller k.
    """
    n, budget = correct.shape
    running = np.zeros(n, dtype=int)
    best_len = np.zeros(n, dtype=int)
    best_end = np.zeros(n, dtype=int)
    for k in range(budget):
        running = np.where(correct[:, k], running + 1, 0)
        better = running > best_len
        best_len = np.where(better, running, best_len)
        best_end = np.where(better, k, best_end)

    run_start = best_end - best_len + 2                 # 1-indexed first k of the run
    middle = run_start + (best_len - 1) // 2
    # A point that is never classified correctly keeps the budget, as the
    # first-correct and fixed-window rules also do.
    k_per_row = np.where(best_len == 0, budget, middle)
    return k_per_row.astype(int), best_len, np.where(best_len == 0, 0, run_start)


def first_correct_from(correct, budget):
    """AdaNN's rule, read off the same matrix."""
    return np.where(correct.any(axis=1), correct.argmax(axis=1) + 1, budget)


# ===========================================================================
# EVALUATION
# ===========================================================================

def prepare_workspace(args):
    os.makedirs(args.workspace, exist_ok=True)
    with open(os.path.join(args.workspace, "config.yaml"), "w", encoding="utf-8") as fh:
        fh.write(CONFIG.format(
            stamp=time.strftime("%Y-%m-%d %H:%M:%S"), base=args.base,
            datasets=json.dumps(args.datasets), backend=args.backend,
            timing="gpu" if args.backend == "gpu" else "cpu",
            folds=args.folds, seed=args.seed))
    sys.argv = ["longest_stable_k", "--workspace", args.workspace]
    import knn_pipeline
    return knn_pipeline


def evaluate(P, blocked_fn, needs_blocking, dataset, model_name, k_max, windows,
             memory_budget, noise=0.0):
    """Every rule, on one dataset, over one shared set of folds."""
    X, y = P.load_pooled_dataset(os.path.join(P.BASE_FOLDER, dataset))
    if X is None:
        return None
    design = P.cv_design(X, y, P.CV_EVAL_FOLDS, P.CV_EVAL_REPEATS, P.CV_SEED)
    if design is None:
        return None
    X, y, splits, n_splits, _ = design
    model_class = P.MODEL_MAP[model_name]

    rows = []
    for repeat, fold, tr, te in splits:
        X_tr, X_te = P.scale_fold(X[tr], X[te])
        y_tr, flipped = P.fold_labels(y[tr], dataset, repeat, fold, noise)
        budget = max(1, min(int(k_max), len(X_tr) - 1))
        blocked = needs_blocking(len(X_tr), memory_budget)

        started = time.time()
        correct = (blocked_fn(P, X_tr, y_tr, model_name, budget) if blocked
                   else P.compute_adaptive_k_last_stable(X_tr, y_tr, model_name,
                                                         budget, return_correctness=True))
        derive = time.time() - started

        centre, run_len, run_start = longest_stable_k(correct)
        # Three parameter-free read-offs of the same longest run. The middle is
        # the original proposal; the start and the end bracket it, and cost
        # nothing extra because the run has already been located. Which one is
        # right is an empirical question: the middle picks a much larger k than
        # any other rule and does not move when labels get noisy, while the
        # start behaves like a short window. Points with no correct k at all
        # keep the budget, as every rule here does.
        no_run = run_len == 0
        start = np.where(no_run, budget, run_start)
        end = np.where(no_run, budget, run_start + run_len - 1)
        candidates = {"LongestStable": centre,
                      "LongestStable_start": start.astype(int),
                      "LongestStable_end": end.astype(int),
                      "AdaNN": first_correct_from(correct, budget)}
        for w in windows:
            candidates[f"Window_w{w}"] = P.scan_stable_window(correct, w)

        for method, per_row in candidates.items():
            predictions, ks = P.predict_with_adaptive_k(model_class, X_tr, y_tr,
                                                        X_te, per_row)
            row = {"Dataset": dataset, "Model": model_name, "Method": method,
                   "Repeat": repeat, "Fold": fold, "k_max": budget,
                   "Accuracy": (predictions == y[te]).mean(),
                   "k_mean": float(np.mean(ks)), "k_std": float(np.std(ks)),
                   "Saturated_frac": float((per_row >= budget).mean()),
                   "N_Train": len(tr), "N_Test": len(te), "Derive_Seconds": derive,
                   "Noise": noise, "Labels_Flipped": flipped}
            if method == "LongestStable":
                row["Run_len_mean"] = float(run_len.mean())
                row["Run_len_max"] = int(run_len.max())
                row["No_run_frac"] = float((run_len == 0).mean())
                row["Run_start_mean"] = float(run_start[run_len > 0].mean()) \
                    if (run_len > 0).any() else np.nan
            rows.append(row)
        print(f"    fold {fold:2d}/{n_splits}  n_train={len(tr):6d}  derive={derive:6.1f}s",
              flush=True)
    return pd.DataFrame(rows)


def run(P, args, outdir):
    from sensitivity_study import correctness_blocked, needs_blocking
    partials = os.path.join(outdir, "partials")
    os.makedirs(partials, exist_ok=True)
    for model in args.models:
        for dataset in args.datasets:
            out = os.path.join(partials, f"{model}_{dataset}.csv")
            if os.path.exists(out):
                print(f"⊘ {model} · {dataset}: already done", flush=True)
                continue
            print(f"▶ {model} · {dataset}", flush=True)
            started = time.time()
            frame = evaluate(P, correctness_blocked, needs_blocking, dataset, model,
                             args.k_max, args.windows, args.memory_budget_gb, args.noise)
            if frame is None or frame.empty:
                print("  ⚠ no result", flush=True)
                continue
            frame.to_csv(out, index=False)
            print(f"  saved in {time.time() - started:.0f}s", flush=True)

    hits = sorted(glob.glob(os.path.join(partials, "*.csv")))
    if not hits:
        return None
    folds = pd.concat([pd.read_csv(h) for h in hits], ignore_index=True)
    folds.to_csv(os.path.join(outdir, "longest_stable_folds.csv"), index=False)
    summary = folds.groupby(["Dataset", "Model", "Method"], as_index=False).agg(
        Accuracy=("Accuracy", "mean"), Std=("Accuracy", "std"),
        k_mean=("k_mean", "mean"), Saturated_frac=("Saturated_frac", "mean"),
        Folds=("Accuracy", "size"))
    summary.to_csv(os.path.join(outdir, "longest_stable_summary.csv"), index=False)
    return folds


# ===========================================================================
# COMPARISON AND FIGURES
# ===========================================================================

def paired_tests(P, folds):
    """LongestStable against each baseline, on the folds they share."""
    rows = []
    for (dataset, model), group in folds.groupby(["Dataset", "Model"]):
        table = group.pivot_table(index=["Repeat", "Fold"], columns="Method",
                                  values="Accuracy")
        if "LongestStable" not in table.columns:
            continue
        n_train, n_test = group["N_Train"].mean(), group["N_Test"].mean()
        for other in [c for c in table.columns if c != "LongestStable"]:
            pair = table[["LongestStable", other]].dropna()
            if len(pair) < 2:
                continue
            diff = (pair["LongestStable"] - pair[other]).values
            _, p = P.corrected_resampled_ttest(diff, n_train, n_test)
            rows.append({"Dataset": dataset, "Model": model, "Baseline": other,
                         "LongestStable": round(pair["LongestStable"].mean(), 4),
                         "Baseline_acc": round(pair[other].mean(), 4),
                         "Delta_pts": round(diff.mean() * 100, 2),
                         "p": round(p, 4) if p == p else np.nan,
                         "Significant": "yes" if (p == p and p < 0.05) else "no",
                         "Wins": "LongestStable" if diff.mean() > 0
                                 else ("tie" if diff.mean() == 0 else other)})
    return pd.DataFrame(rows)


def style(ax):
    ax.set_facecolor(BG)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure(folds, tests, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = folds.groupby(["Dataset", "Model", "Method"], as_index=False).agg(
        Accuracy=("Accuracy", "mean"), Std=("Accuracy", "std"))
    models = [m for m in MODELS if m in summary["Model"].unique()]
    fig, axes = plt.subplots(2, len(models), figsize=(5.2 * len(models), 8.2),
                             squeeze=False)
    fig.patch.set_facecolor(BG)

    for col, model in enumerate(models):
        part = summary[summary["Model"] == model]
        grid = part.pivot_table(index="Dataset", columns="Method", values="Accuracy")
        methods = [m for m in ["AdaNN"] + sorted(c for c in grid.columns
                                                 if c.startswith("Window"))
                   + ["LongestStable"] if m in grid.columns]
        # Top: how far each baseline sits from the new rule, dataset by dataset.
        ax = axes[0][col]
        gaps = {m: (grid["LongestStable"] - grid[m]) * 100 for m in methods
                if m != "LongestStable"}
        x = np.arange(len(grid.index))
        width = 0.8 / max(1, len(gaps))
        for i, (name, values) in enumerate(gaps.items()):
            ax.bar(x + i * width - 0.4 + width / 2, values, width * 0.9, label=name)
        ax.axhline(0, color="#111", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(grid.index, fontsize=8, rotation=45, ha="right")
        ax.set_ylabel("LongestStable minus baseline\n(points)")
        ax.set_title(LABEL[model], fontsize=11, loc="left")
        style(ax)
        ax.legend(frameon=False, fontsize=8)

        # Bottom: the k it chooses, and how wide the stable band was.
        ax = axes[1][col]
        chosen = folds[(folds["Model"] == model) & (folds["Method"] == "LongestStable")]
        by_dataset = chosen.groupby("Dataset")[["k_mean", "Run_len_mean"]].mean()
        by_dataset = by_dataset.reindex(grid.index)
        ax.plot(x, by_dataset["k_mean"], "o-", color=COLOUR[model], label="chosen k")
        ax.plot(x, by_dataset["Run_len_mean"], "s--", color="#888",
                label="length of the stable run")
        ax.set_xticks(x)
        ax.set_xticklabels(grid.index, fontsize=8, rotation=45, ha="right")
        ax.set_ylabel("mean over training points")
        style(ax)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Longest stable run, centred — against the fixed-window and "
                 "first-correct rules", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150, facecolor=BG)
    plt.close(fig)


def write_table(frame, outdir, stem, caption):
    if frame is None or frame.empty:
        return
    frame = frame.where(pd.notna(frame), "—")
    frame.to_csv(os.path.join(outdir, f"{stem}.csv"), index=False)
    with open(os.path.join(outdir, f"{stem}.tex"), "w", encoding="utf-8") as fh:
        fh.write(frame.to_latex(index=False, escape=True, longtable=len(frame) > 40,
                                caption=caption, label=f"tab:{stem}"))
    print(f"  {stem}.csv / .tex   ({len(frame)} rows)")


def report(P, folds, outdir):
    tests = paired_tests(P, folds)
    figure(folds, tests, os.path.join(outdir, "fig_longest_stable.png"))
    print("  fig_longest_stable.png")
    write_table(tests, outdir, "table_longest_stable_pairs",
                "The centred longest-run rule against each baseline, paired over "
                "identical folds (Nadeau \\& Bengio corrected resampled $t$-test).")

    overall = []
    for (model, baseline), group in tests.groupby(["Model", "Baseline"]):
        overall.append({
            "Model": LABEL.get(model, model), "Baseline": baseline,
            "Datasets": len(group),
            "Mean_delta_pts": round(group["Delta_pts"].mean(), 2),
            "Wins": int((group["Delta_pts"] > 0).sum()),
            "Significant_for": int(((group["Significant"] == "yes") &
                                    (group["Delta_pts"] > 0)).sum()),
            "Significant_against": int(((group["Significant"] == "yes") &
                                        (group["Delta_pts"] < 0)).sum())})
    write_table(pd.DataFrame(overall), outdir, "table_longest_stable_overview",
                "Summary of the centred longest-run rule against each baseline.")

    print("\n" + "=" * 74)
    for row in overall:
        print(f"{row['Model']:16s} vs {row['Baseline']:14s} "
              f"Δ={row['Mean_delta_pts']:+6.2f} pts   "
              f"wins {row['Wins']}/{row['Datasets']}   "
              f"significant {row['Significant_for']} for / "
              f"{row['Significant_against']} against")
    print("=" * 74)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default="longest_stable_run")
    ap.add_argument("--base", default="datasets")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--backend", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--k-max", type=int, default=30)
    ap.add_argument("--windows", nargs="+", type=int, default=[3],
                    help="fixed windows to compare against")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--memory-budget-gb", type=float, default=8.0)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="fraction of each fold's TRAINING labels to flip, e.g. 0.3; "
                         "test labels are never touched")
    ap.add_argument("--figures-only", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = os.path.join(args.workspace, "results")
    os.makedirs(outdir, exist_ok=True)
    P = prepare_workspace(args)

    print("=" * 74)
    print(f"workspace : {args.workspace}")
    print(f"datasets  : {' '.join(args.datasets)}")
    print(f"models    : {' '.join(args.models)}")
    print(f"protocol  : {P.CV_EVAL_FOLDS}-fold, seed {P.CV_SEED}, scaling={P.CV_SCALING}")
    print(f"backend   : {P.get_backend()}    k_max = {args.k_max}")
    print(f"compared  : AdaNN, " + ", ".join(f"Window_w{w}" for w in args.windows))
    print(f"label noise: " + (f"{args.noise*100:g}% of each fold's TRAINING labels"
                              if args.noise else "none (clean data)"))
    print("=" * 74, flush=True)

    if args.figures_only:
        path = os.path.join(outdir, "longest_stable_folds.csv")
        if not os.path.exists(path):
            print("nothing computed yet")
            return
        folds = pd.read_csv(path)
    else:
        started = time.time()
        folds = run(P, args, outdir)
        if folds is None:
            print("nothing computed")
            return
        print(f"\nfinished in {(time.time() - started) / 60:.1f} min", flush=True)

    report(P, folds, outdir)
    print(f"\nwritten to {outdir}/")


if __name__ == "__main__":
    main()
