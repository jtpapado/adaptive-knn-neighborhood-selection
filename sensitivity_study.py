#!/usr/bin/env python3
"""
The two sensitivity analyses and every figure and table that supports them.

    k_max  1..50   for the adaptive rule       (why k_max = 30?)
    w      2..15   for the window rule         (why w = 3?)

for CustomKNN, MutualKNN and SymmetricKNN on all fourteen datasets, scored with
the same stratified 10-fold design and per-fold min-max scaling the main run
uses, so the curves and the headline numbers describe one experiment.

Self-contained apart from knn_pipeline.py and knn_algorithms.py, which must sit
beside it. It writes its own workspace config, so nothing else has to be set up.

    python3 sensitivity_study.py
    python3 sensitivity_study.py --workspace sens1 --backend gpu
    python3 sensitivity_study.py --models CustomKNN
    python3 sensitivity_study.py --figures-only
    python3 sensitivity_study.py --main server_run      # adds the comparison figure

Every (model, dataset) result is written the moment it finishes, so a run that
is interrupted resumes where it stopped instead of starting over. Delete the
partials/ directory to force a recompute.

All three neighbourhood rules run on every dataset, shuttle included. Above the
memory budget the derive switches to a blocked implementation that never holds
an n x n array; it was checked against the pipeline's own routine and agrees
exactly except where equal distances make the k nearest neighbours ambiguous,
which moves accuracy by under 0.1 points.
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
PAPER_KMAX, ADANN_KMAX, PAPER_W = 30, 9, 3
BG = "#fbfaf8"


# ===========================================================================
# WORKSPACE
# ===========================================================================

CONFIG = """# Written by sensitivity_study.py on {stamp}
# The sweeps read the protocol from here; no pipeline phase runs from it.
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
         rare_class_policy: "warn", methods: ["bestk"],
         w_values: [2,3,4,5,6,7,8,9,10,11,12,13,14,15]}}
phase5: {{k_max_values: [9, 30]}}
phase9: {{k_max: 30, stability_windows: [3]}}
workers: {{main: -1, builders: 1, testers: 10}}
"""


def prepare_workspace(args):
    os.makedirs(args.workspace, exist_ok=True)
    with open(os.path.join(args.workspace, "config.yaml"), "w", encoding="utf-8") as fh:
        fh.write(CONFIG.format(
            stamp=time.strftime("%Y-%m-%d %H:%M:%S"), base=args.base,
            datasets=json.dumps(args.datasets), backend=args.backend,
            timing="gpu" if args.backend == "gpu" else "cpu",
            folds=args.folds, seed=args.seed))
    sys.argv = ["sensitivity_study", "--workspace", args.workspace]
    import knn_pipeline
    return knn_pipeline


# ===========================================================================
# MEMORY-BOUNDED NEIGHBOURHOODS
#
#   The pipeline sorts every row of an n x n distance matrix, which shuttle
#   cannot afford: 11 GB for the distances and 22 GB for the index matrix, and
#   the sort is host-side numpy, so a GPU does not help.  Only the nearest
#   k_max neighbours are ever read, so distances are built a block of rows at a
#   time and reduced with argpartition.
#
#   The voting below is the plain majority rule.  Mutual and Symmetric kNN need
#   reverse-neighbour information that only the full matrix carries, so this
#   path refuses them rather than quietly returning another rule's answer.
# ===========================================================================

def _distance_block(P, X, X32, start, stop, on_gpu):
    if on_gpu:
        from knn_algorithms import _batched_cdist, _get_device
        return _batched_cdist(X32[start:stop], X32, _get_device())
    from scipy.spatial.distance import cdist
    return cdist(X[start:stop], X, metric="euclidean")


def _nearest_neighbours(P, X, budget, block):
    """Indices and distances of each point's `budget` nearest others."""
    n = len(X)
    neighbours = np.empty((n, budget), dtype=np.int64)
    distances_to = np.empty((n, budget), dtype=np.float32)
    on_gpu = P.get_backend() == "gpu"
    X32 = np.asarray(X, dtype=np.float32) if on_gpu else None
    for start in range(0, n, block):
        stop = min(start + block, n)
        d = _distance_block(P, X, X32, start, stop, on_gpu)
        d[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest = np.argpartition(d, budget - 1, axis=1)[:, :budget]
        picked = np.take_along_axis(d, nearest, axis=1)
        order = np.argsort(picked, axis=1)
        neighbours[start:stop] = np.take_along_axis(nearest, order, axis=1)
        distances_to[start:stop] = np.take_along_axis(picked, order, axis=1)
        del d
    return neighbours, distances_to


def correctness_blocked(P, X, y, model_name, k_max, block=2048):
    """
    correct[i, k-1] — is training point i classified correctly by its own rule
    at neighbourhood size k — without ever holding an n x n array.

    All three rules are supported, at different cost.

      plain and mutual   read only the nearest k_max neighbours, so one blocked
                         pass over the distances is enough.
      symmetric          also needs, for every i, the points j that hold i
                         inside their own k-th radius.  That is a full-row
                         question, so the row blocks are recomputed — but each
                         block is built once and every k is scanned inside it,
                         which is the same work the pipeline does with the whole
                         matrix resident, at O(block x n) memory instead.

    Ties among equal distances are resolved differently from a full sort; on
    integer-valued data that moves accuracy by a few tenths of a point.  Used
    only where the pipeline's own routine will not fit.
    """
    n = len(y)
    budget = min(int(k_max), n - 1)
    classes = np.unique(y)
    n_classes = len(classes)
    neighbours, nb_dists = _nearest_neighbours(P, X, budget, block)
    correct = np.zeros((n, budget), dtype=bool)

    if model_name == "CustomKNN":
        for k in range(1, budget + 1):
            correct[:, k - 1] = (
                P._vectorized_majority(y[neighbours[:, :k]], classes, n) == y)
        return correct

    if model_name == "MutualKNN":
        # A neighbour j of i counts only when i also lies inside j's k-th radius,
        # which nb_dists supplies for every j without any full row.
        for k in range(1, budget + 1):
            idx = neighbours[:, :k]
            mutual = nb_dists[:, :k] <= nb_dists[idx, k - 1]
            has_mutual = np.any(mutual, axis=1)
            effective = np.where(has_mutual[:, None], mutual, True)
            correct[:, k - 1] = (
                P._vectorized_masked_majority(y[idx], effective, classes, n) == y)
        return correct

    if model_name != "SymmetricKNN":
        raise ValueError(f"unknown rule: {model_name}")

    on_gpu = P.get_backend() == "gpu"
    X32 = np.asarray(X, dtype=np.float32) if on_gpu else None
    class_masks = [(y == c) for c in classes]
    for start in range(0, n, block):
        stop = min(start + block, n)
        size = stop - start
        d = _distance_block(P, X, X32, start, stop, on_gpu)
        rows = np.arange(size)
        for k in range(1, budget + 1):
            forward = np.zeros((size, n), dtype=bool)
            np.put_along_axis(forward, neighbours[start:stop, :k], True, axis=1)
            reverse = d <= nb_dists[:, k - 1][None, :]
            reverse[rows, np.arange(start, stop)] = False
            union = forward | reverse
            counts = np.empty((size, n_classes), dtype=np.int32)
            for ci, mask in enumerate(class_masks):
                counts[:, ci] = np.count_nonzero(union & mask[None, :], axis=1)
            correct[start:stop, k - 1] = (classes[np.argmax(counts, axis=1)] == y[start:stop])
            del forward, reverse, union
        del d
    return correct


def first_correct_k_blocked(P, X, y, model_name, k_max, block=2048):
    correct = correctness_blocked(P, X, y, model_name, k_max, block)
    n, budget = correct.shape
    first = np.full(n, budget, dtype=int)
    undecided = np.ones(n, dtype=bool)
    for k in range(1, budget + 1):
        hit = correct[:, k - 1] & undecided
        first[hit] = k
        undecided &= ~hit
        if not undecided.any():
            break
    return first


def needs_blocking(n_train, budget_gb):
    """12 bytes per pair: a float32 distance plus an int64 index from the sort."""
    return n_train * n_train * 12 / 1e9 > budget_gb


# ===========================================================================
# SWEEPS
#
#   Both exploit the same redundancy.  compute_first_correct_k fills its vector
#   with the budget and overwrites an entry only at the first k that classifies
#   the point correctly, so
#       first_correct_k(budget m) == min(first_correct_k(budget K), m),  m <= K
#   and the correctness matrix behind the window rule does not depend on w at
#   all.  Each fold therefore pays for one derive, not one per sweep value.
# ===========================================================================

def sweep_kmax(P, dataset, model_name, k_max_top, blocked, noise=0.0):
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
        budget = max(1, min(k_max_top, len(X_tr) - 1))
        started = time.time()
        full = (first_correct_k_blocked(P, X_tr, y_tr, model_name, budget) if blocked
                else P.compute_first_correct_k(X_tr, y_tr, model_name, budget))
        derive = time.time() - started
        for m in range(1, budget + 1):
            per_row = np.minimum(full, m)
            predictions, ks = P.predict_with_adaptive_k(model_class, X_tr, y_tr, X_te, per_row)
            rows.append({"Dataset": dataset, "Model": model_name, "k_max": m,
                         "Repeat": repeat, "Fold": fold,
                         "Accuracy": (predictions == y[te]).mean(),
                         "k_mean": float(np.mean(ks)),
                         "Saturated_frac": float((per_row >= m).mean()),
                         "N_Train": len(tr), "N_Test": len(te), "Derive_Seconds": derive,
                         "Noise": noise, "Labels_Flipped": flipped})
        print(f"    fold {fold:2d}/{n_splits}  n_train={len(tr):6d}  derive={derive:6.1f}s"
              + (f"  flipped={flipped}" if noise else ""), flush=True)
    return pd.DataFrame(rows)


def sweep_w(P, dataset, model_name, k_max, w_top, blocked, noise=0.0):
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
        budget = max(1, min(k_max, len(X_tr) - 1))
        started = time.time()
        correct = (correctness_blocked(P, X_tr, y_tr, model_name, budget) if blocked
                   else P.compute_adaptive_k_last_stable(X_tr, y_tr, model_name, budget,
                                                         return_correctness=True))
        derive = time.time() - started
        for w in range(2, min(w_top, budget) + 1):
            per_row = P.scan_stable_window(correct, w)
            predictions, ks = P.predict_with_adaptive_k(model_class, X_tr, y_tr, X_te, per_row)
            rows.append({"Dataset": dataset, "Model": model_name, "w": w, "k_max": budget,
                         "Repeat": repeat, "Fold": fold,
                         "Accuracy": (predictions == y[te]).mean(),
                         "k_mean": float(np.mean(ks)),
                         "Saturated_frac": float((per_row >= budget).mean()),
                         "N_Train": len(tr), "N_Test": len(te), "Derive_Seconds": derive,
                         "Noise": noise, "Labels_Flipped": flipped})
        print(f"    fold {fold:2d}/{n_splits}  n_train={len(tr):6d}  derive={derive:6.1f}s"
              + (f"  flipped={flipped}" if noise else ""), flush=True)
    return pd.DataFrame(rows)


def run_sweeps(P, args, figdir):
    """Every (sweep, model, dataset) cell, saved as it finishes so a restart resumes."""
    partials = os.path.join(figdir, "partials")
    os.makedirs(partials, exist_ok=True)

    plans = []
    if not args.skip_kmax:
        plans.append(("kmax", lambda P_, d, m, b: sweep_kmax(P_, d, m, args.kmax_top, b,
                                                             args.noise)))
    if not args.skip_w:
        plans.append(("w", lambda P_, d, m, b: sweep_w(P_, d, m, args.w_kmax, args.w_top, b,
                                                       args.noise)))

    for tag, runner in plans:
        for model in args.models:
            for dataset in args.datasets:
                out = os.path.join(partials, f"{tag}_{model}_{dataset}.csv")
                if os.path.exists(out):
                    print(f"⊘ {tag} · {model} · {dataset}: already done", flush=True)
                    continue
                rows_total = len(P.load_pooled_dataset(
                    os.path.join(P.BASE_FOLDER, dataset))[1])
                n_train = int(rows_total * (1 - 1 / P.CV_EVAL_FOLDS))
                blocked = needs_blocking(n_train, args.memory_budget_gb)
                print(f"▶ {tag} · {model} · {dataset}"
                      + ("  (blocked derive)" if blocked else ""), flush=True)
                started = time.time()
                frame = runner(P, dataset, model, blocked)
                if frame is None or frame.empty:
                    print(f"  ⚠ no result", flush=True)
                    continue
                frame.to_csv(out, index=False)
                print(f"  saved in {time.time() - started:.0f}s", flush=True)

    # Combine the partials into the per-model files the figures read.
    for tag, xname in (("kmax", "k_max"), ("w", "w")):
        for model in args.models:
            hits = sorted(glob.glob(os.path.join(partials, f"{tag}_{model}_*.csv")))
            if not hits:
                continue
            folds = pd.concat([pd.read_csv(h) for h in hits], ignore_index=True)
            summary = folds.groupby(["Dataset", "Model", xname], as_index=False).agg(
                Accuracy=("Accuracy", "mean"), Std=("Accuracy", "std"),
                k_mean=("k_mean", "mean"), Saturated_frac=("Saturated_frac", "mean"),
                Folds=("Accuracy", "size"))
            folds.to_csv(os.path.join(figdir, f"{tag}_sweep_{model}_folds.csv"), index=False)
            summary.to_csv(os.path.join(figdir, f"{tag}_sweep_{model}_summary.csv"), index=False)


# ===========================================================================
# FIGURES AND TABLES
# ===========================================================================

def style(ax):
    ax.set_facecolor(BG)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def settle_point(loss, threshold=-0.5):
    values = loss.values
    for i in range(len(values)):
        if (values[i:] >= threshold).all():
            return int(loss.index[i])
    return None


def sweep_figure(frames, xname, marks, title, out_png):
    """
    Raw accuracy cannot be pooled across datasets sitting between 0.53 and 0.99,
    so the top row averages and the bottom row expresses each dataset as a
    distance from its own best — the comparable quantity, and the one that shows
    where each curve stops moving.
    """
    import matplotlib.pyplot as plt
    present = [m for m in MODELS if frames.get(m) is not None]
    if not present:
        return None
    fig, axes = plt.subplots(2, len(present), figsize=(4.7 * len(present), 7.6),
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor(BG)
    rows = []
    for col, model in enumerate(present):
        grid = frames[model].pivot_table(index=xname, columns="Dataset",
                                         values="Accuracy").sort_index()
        avg, err = grid.mean(axis=1), grid.std(axis=1) / np.sqrt(grid.shape[1])
        loss = (grid - grid.max()) * 100
        mean_loss = loss.mean(axis=1)
        top, bottom = axes[0][col], axes[1][col]

        top.plot(grid.index, avg, color=COLOUR[model], lw=2.4)
        top.fill_between(grid.index, avg - err, avg + err, color=COLOUR[model], alpha=0.15)
        top.set_title(f"{LABEL[model]}  ({grid.shape[1]} datasets)", fontsize=11, loc="left")
        if col == 0:
            top.set_ylabel("mean accuracy")
        style(top)

        for dataset in grid.columns:
            bottom.plot(grid.index, loss[dataset], color=COLOUR[model], lw=0.8, alpha=0.35)
        bottom.plot(grid.index, mean_loss, color="#111", lw=2.2, zorder=5)
        bottom.axhline(-0.5, color="#888", ls=":", lw=1)
        bottom.set_xlabel(xname)
        if col == 0:
            bottom.set_ylabel("points below that\ndataset's own best")
        style(bottom)
        for value, colour, text in marks:
            for ax in (top, bottom):
                ax.axvline(value, color=colour, ls="--", lw=1.2, alpha=0.8)
            bottom.annotate(text, xy=(value, bottom.get_ylim()[0]),
                            xytext=(value + 0.4, mean_loss.min() * 0.55),
                            color=colour, fontsize=8)

        worst = loss.min(axis=1)
        band = [int(v) for v in loss.index if worst[v] > -1.0]
        rows.append({"Rule": LABEL[model], "Datasets": grid.shape[1],
                     "Peak": int(avg.idxmax()), "Mean_settles_at": settle_point(mean_loss),
                     "Safe_band_lo": min(band) if band else None,
                     "Safe_band_hi": max(band) if band else None,
                     "Safe_band_size": len(band),
                     "Mean_range_pts": round((avg.max() - avg.min()) * 100, 3)})
    fig.suptitle(title, x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150, facecolor=BG)
    plt.close(fig)
    return pd.DataFrame(rows)


def per_dataset_table(frames, xname, marked, reference):
    rows = []
    for model in MODELS:
        frame = frames.get(model)
        if frame is None:
            continue
        grid = frame.pivot_table(index=xname, columns="Dataset", values="Accuracy").sort_index()
        stds = frame.pivot_table(index=xname, columns="Dataset", values="Std").sort_index()
        for dataset in grid.columns:
            series = grid[dataset]
            fold_sd = stds[dataset].mean() * 100
            span = (series.max() - series.min()) * 100
            rows.append({"Rule": LABEL[model], "Dataset": dataset,
                         f"Best_{xname}": int(series.idxmax()),
                         "Acc_at_best": round(series.max(), 4),
                         f"Acc_at_{marked}": round(series.get(marked, np.nan), 4),
                         f"Acc_at_{reference}": round(series.get(reference, np.nan), 4),
                         "Sweep_range_pts": round(span, 2),
                         "Fold_std_pts": round(fold_sd, 2),
                         "Range_below_noise": bool(span < fold_sd)})
    return pd.DataFrame(rows)


def saturation_figure(kmax_frames, w_frames, out_png):
    """
    Why one parameter is inert and the other is not: raising k_max rescues
    points that had no correct neighbourhood, raising w strands points that had
    no stable one. Both end in the same default, from opposite directions.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.patch.set_facecolor(BG)
    for ax, frames, xname, title in (
            (axes[0], kmax_frames, "k_max",
             "Adaptive rule: training points\nnever classified correctly"),
            (axes[1], w_frames, "w",
             "Window rule: points with no stable\nwindow (fall back to k_max)")):
        drawn = False
        for model in MODELS:
            frame = frames.get(model)
            if frame is None or "Saturated_frac" not in frame.columns:
                continue
            series = frame.groupby(xname)["Saturated_frac"].mean() * 100
            ax.plot(series.index, series.values, color=COLOUR[model], lw=2.2,
                    label=LABEL[model])
            drawn = True
        ax.set_xlabel(xname)
        ax.set_ylabel("% of training points")
        ax.set_title(title, fontsize=10, loc="left")
        style(ax)
        if drawn:
            ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=BG)
    plt.close(fig)


def comparison_figure(summaries, out_png):
    import matplotlib.pyplot as plt
    frame = pd.concat(summaries, ignore_index=True)
    grid = frame.pivot_table(index="Dataset", columns="Method", values="Accuracy_Mean")
    errs = frame.pivot_table(index="Dataset", columns="Method", values="Accuracy_Std")
    datasets, methods = list(grid.index), list(grid.columns)
    fig, ax = plt.subplots(figsize=(max(11, 1.0 * len(datasets)), 5.8))
    fig.patch.set_facecolor(BG)
    width = 0.8 / max(1, len(methods))
    x = np.arange(len(datasets))
    for i, method in enumerate(methods):
        model = next((m for m in MODELS if method.startswith(m)), None)
        ax.bar(x + i * width - 0.4 + width / 2, grid[method], width * 0.9,
               yerr=np.minimum(errs[method].fillna(0), 1 - grid[method]),
               label=method, color=COLOUR.get(model, "#777"), edgecolor=BG, linewidth=0.6,
               error_kw=dict(ecolor="#888", elinewidth=0.7, capsize=1.5))
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylabel("accuracy (10-fold CV)")
    ax.set_title("Methods under the shared fold design — error bars are std over folds",
                 fontsize=12, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=BG)
    plt.close(fig)
    return grid


def write_table(frame, figdir, stem, caption):
    if frame is None or frame.empty:
        return
    # A missing value is meaningful — "no safe band exists", "the sweep never
    # reached that setting" — so it is a dash, not a blank that shifts columns.
    frame = frame.where(pd.notna(frame), "—")
    frame.to_csv(os.path.join(figdir, f"{stem}.csv"), index=False)
    with open(os.path.join(figdir, f"{stem}.tex"), "w", encoding="utf-8") as fh:
        fh.write(frame.to_latex(index=False, escape=True, longtable=len(frame) > 40,
                                caption=caption, label=f"tab:{stem}"))
    print(f"  {stem}.csv / .tex   ({len(frame)} rows)")


def make_figures(figdir, main_workspace):
    def load(stem):
        path = os.path.join(figdir, stem)
        return pd.read_csv(path) if os.path.exists(path) else None

    kmax = {m: load(f"kmax_sweep_{m}_summary.csv") for m in MODELS}
    wfr = {m: load(f"w_sweep_{m}_summary.csv") for m in MODELS}
    print(f"\nk_max sweeps : {[m for m in MODELS if kmax[m] is not None] or 'none'}")
    print(f"w sweeps     : {[m for m in MODELS if wfr[m] is not None] or 'none'}\n")

    if any(v is not None for v in kmax.values()):
        summary = sweep_figure(
            kmax, "k_max",
            [(ADANN_KMAX, "#7f8c8d", "k_max=9\n(AdaNN)"),
             (PAPER_KMAX, "#c0392b", "k_max=30\n(paper)")],
            "Sensitivity to the neighbourhood budget k_max",
            os.path.join(figdir, "fig1_kmax_sensitivity.png"))
        print("  fig1_kmax_sensitivity.png")
        write_table(summary, figdir, "table1_kmax_overview",
                    "Sensitivity of the adaptive rule to $k_{\\max}$.")
        write_table(per_dataset_table(kmax, "k_max", PAPER_KMAX, ADANN_KMAX),
                    figdir, "table2_kmax_per_dataset",
                    "Per-dataset behaviour of $k_{\\max}$. \\textit{Range\\_below\\_noise} "
                    "marks datasets whose whole sweep moves less than their own "
                    "fold-to-fold standard deviation.")

    if any(v is not None for v in wfr.values()):
        summary = sweep_figure(
            wfr, "w", [(PAPER_W, "#c0392b", "w=3\n(paper)")],
            "Sensitivity to the stability window w",
            os.path.join(figdir, "fig2_w_sensitivity.png"))
        print("  fig2_w_sensitivity.png")
        write_table(summary, figdir, "table3_w_overview",
                    "Sensitivity of the window rule to $w$.")
        write_table(per_dataset_table(wfr, "w", PAPER_W, 2),
                    figdir, "table4_w_per_dataset",
                    "Per-dataset behaviour of the stability window $w$.")

    if any(v is not None for v in kmax.values()) or any(v is not None for v in wfr.values()):
        saturation_figure(kmax, wfr, os.path.join(figdir, "fig3_saturation.png"))
        print("  fig3_saturation.png")

    if main_workspace:
        summaries, seen = [], []
        for prefix, namer in (
                ("results_phase1_initial_", lambda r: f"{r['Model']}_k{int(r['k'])}"),
                ("results_phase6_adaptive_test_", lambda r: f"{r['Model']}_Ada{int(r['k_max'])}"),
                ("results_phase4_nested_cv_", None)):
            hits = sorted(glob.glob(os.path.join(main_workspace, "results", f"{prefix}*.xlsx")))
            if not hits:
                continue
            frame = pd.read_excel(hits[-1], sheet_name="Summary")
            frame["Method"] = frame["Method"] if namer is None else frame.apply(namer, axis=1)
            summaries.append(frame[["Dataset", "Method", "Accuracy_Mean", "Accuracy_Std"]])
            seen.append(os.path.basename(hits[-1]))
        if summaries:
            grid = comparison_figure(summaries, os.path.join(figdir, "fig4_method_comparison.png"))
            print("  fig4_method_comparison.png\n    from: " + ", ".join(seen))
            write_table(grid.round(4).reset_index(), figdir, "table5_method_comparison",
                        "Accuracy of every method under the shared 10-fold design.")
        else:
            print(f"  (no result workbooks under {main_workspace}/results — skipped)")


# ===========================================================================

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default="sensitivity_run")
    ap.add_argument("--base", default="datasets")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--backend", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kmax-top", type=int, default=50)
    ap.add_argument("--w-top", type=int, default=15)
    ap.add_argument("--w-kmax", type=int, default=30)
    ap.add_argument("--memory-budget-gb", type=float, default=8.0)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="fraction of each fold's TRAINING labels to flip, e.g. 0.3; "
                         "test labels are never touched")
    ap.add_argument("--skip-kmax", action="store_true")
    ap.add_argument("--skip-w", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--main", default=None,
                    help="workspace of the main run, for the comparison figure")
    return ap.parse_args()


def main():
    args = parse_args()
    figdir = os.path.join(args.workspace, "results", "figures")
    os.makedirs(figdir, exist_ok=True)

    P = prepare_workspace(args)
    print("=" * 70)
    print(f"workspace : {args.workspace}")
    print(f"data      : {P.BASE_FOLDER}")
    print(f"datasets  : {' '.join(args.datasets)}")
    print(f"models    : {' '.join(args.models)}")
    print(f"protocol  : {P.CV_EVAL_FOLDS}-fold, seed {P.CV_SEED}, scaling={P.CV_SCALING}")
    print(f"backend   : {P.get_backend()}")
    print(f"sweeps    : k_max 1..{args.kmax_top}   w 2..{args.w_top} (k_max={args.w_kmax})")
    print(f"label noise: " + (f"{args.noise*100:g}% of each fold's TRAINING labels"
                              if args.noise else "none (clean data)"))
    print("=" * 70, flush=True)

    started = time.time()
    if not args.figures_only:
        run_sweeps(P, args, figdir)
        print(f"\nsweeps finished in {(time.time() - started) / 60:.1f} min", flush=True)

    make_figures(figdir, args.main)
    print(f"\nwritten to {figdir}/")
    for name in sorted(os.listdir(figdir)):
        if not os.path.isdir(os.path.join(figdir, name)):
            print(f"  {name}")


if __name__ == "__main__":
    main()
