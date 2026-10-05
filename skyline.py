#!/usr/bin/env python3
"""
Accuracy-efficiency skyline for the six methods on three neighbourhood rules.

Where each time comes from:

    k = 5        Phase 1, summed over folds
    Best k       Phase 4, which includes the inner tuning search
    Ada 9 / 30   Phase 5 (deriving the neighbourhoods) plus Phase 6 (scoring)
    WAda / LAda  the window run, which scores four rules off one correctness
                 matrix and logs one wall time per (rule, dataset). The matrix
                 is charged in full to each method, because either would have to
                 build it alone; the remaining time is split evenly between the
                 four. This is the one estimated quantity in the figure, and it
                 only affects the scoring part, which is the smaller half.
    WAda w = 10  accuracy from the window sweep, the run that covers w = 10 on
                 all fourteen datasets; cost as for w = 3, since every window is
                 read off the same correctness matrix.

All timings were measured in the same GPU runs, so the axis is comparable.
"""
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = ["mgt", "pd", "ls", "ph", "bn", "ys", "pm", "txr", "sh", "ecl", "iris", "tn", "bl", "lir"]
RULES = [("CustomKNN", "Directed", "#2f6fdb"), ("MutualKNN", "Mutual", "#e8703a"),
         ("SymmetricKNN", "Symmetric", "#1faa72")]
MARK = {"BestK": "o", "k5": "s", "Ada9": "^", "Ada30": "v", "WAda3": "D",
        "WAda10": "X", "WAdaT": "P", "LSW": "*"}

# Legend text: the family, with no rule attached.
SHORT = {"BestK": "Best $k$", "k5": "$k$=5", "Ada9": "Ada$_9$",
         "Ada30": "Ada$_{30}$", "WAda3": "WAda$_{w=3}$",
         "WAda10": "WAda$_{w=10}$", "WAdaT": "WAda$_{w^*}$", "LSW": "LAda$_{30}$"}

# Point text: the name the method carries in the manuscript. Ada/WAda take the
# suffix of their neighbourhood rule (NN, MNN, SNN); LAda is the longest-stable
# variant and follows the same pattern.
STEM = {"Directed": "NN", "Mutual": "MNN", "Symmetric": "SNN"}


def label(rule, method):
    if method == "BestK":
        return "Best $k$"
    if method == "k5":
        return "$k$=5"
    stem = STEM[rule]
    if method == "Ada9":
        return f"Ada{stem}$_9$"
    if method == "Ada30":
        return f"Ada{stem}$_{{30}}$"
    if method == "WAda3":
        return f"WAda{stem}$_{{3}}$"
    if method == "WAda10":
        return f"WAda{stem}$_{{10}}$"
    if method == "WAdaT":
        return f"WAda{stem}$_{{w^*}}$"
    return f"LAda{stem}$_{{30}}$"


BG = "#fbfaf8"

# The symmetric cluster piles up near the right edge, and in each panel a
# different pair of points lands close enough to overprint. Nudge by hand.
NUDGE = {
    "clean": {
        ("Symmetric", "Ada9"): (7, -12),
        ("Symmetric", "Ada30"): (-8, -14),
        ("Symmetric", "WAda3"): (-16, 9),
        ("Directed", "Ada30"): (6, -12),
        ("Symmetric", "LSW"): (-9, 5),
        ("Directed", "WAdaT"): (8, -12),
    },
    "noise30": {
        ("Symmetric", "Ada9"): (7, -12),
        ("Symmetric", "Ada30"): (-8, -14),
        ("Symmetric", "WAda3"): (-16, 9),
        ("Directed", "Ada9"): (6, -12),
        ("Mutual", "WAda3"): (8, 6),
        ("Mutual", "LSW"): (8, -12),
    },
}


def newest(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def log_walltimes(paths):
    """One wall-clock second count per (rule, dataset), as the run reported it."""
    out, current = {}, None
    for path in paths:
        for line in open(path, errors="replace"):
            m = re.match(r"^▶ (\w+) · (\w+)", line.strip())
            if m:
                current = (m.group(1), m.group(2))
                continue
            m = re.match(r"^\s*saved in (\d+)s", line)
            if m and current:
                out[current] = int(m.group(1))
                current = None
    return out


def gather(root, noisy):
    pipe = ["srv_noise30b"] if noisy else ["srv_gpu", "srv_gpu_sh"]
    sub = "results/noise30_seed42" if noisy else "results"
    acc, sec = {}, {}

    def add(key, a, s):
        acc[key] = a
        sec[key] = s

    for ws in pipe:
        base = os.path.join(root, "server_results", ws, sub)
        f = newest(os.path.join(base, "results_phase1_initial_*.xlsx"))
        if f:
            for _, r in pd.read_excel(f, sheet_name="Summary").iterrows():
                add((r["Model"], r["Dataset"], "k5"), r["Accuracy_Mean"], r["Time_Total"])
        f = newest(os.path.join(base, "results_phase4_nested_cv_*.xlsx"))
        if f:
            for _, r in pd.read_excel(f, sheet_name="Summary").iterrows():
                if r["Method"].endswith("_BestK"):
                    add((r["Method"][:-6], r["Dataset"], "BestK"),
                        r["Accuracy_Mean"], r["Time_Total"])
                elif r["Method"].endswith("_LastK_tunedW"):
                    add((r["Method"][:-13], r["Dataset"], "WAdaT"),
                        r["Accuracy_Mean"], r["Time_Total"])
        f = newest(os.path.join(base, "results_phase6_adaptive_test_*.xlsx"))
        score = {}
        if f:
            for _, r in pd.read_excel(f, sheet_name="Summary").iterrows():
                score[(r["Model"], r["Dataset"], int(r["k_max"]))] = (r["Accuracy_Mean"],
                                                                     r["Time_Total"])
        derive = {}
        for c in glob.glob(os.path.join(base, "results_phase5_adaptive_timing_*.csv")):
            d = pd.read_csv(c)
            tcol = next((x for x in d.columns if "Time" in x), None)
            for _, r in d.iterrows():
                derive[(r["Model"], r["Dataset"], int(r["k_max"]))] = r[tcol]
        for key, (a, s) in score.items():
            add((key[0], key[1], f"Ada{key[2]}"), a, s + derive.get(key, 0.0))

    # the window family
    if noisy:
        folds = pd.read_csv(os.path.join(root, "server_results", "ls_noise30",
                                         "results", "longest_stable_folds.csv"))
        walls = log_walltimes([os.path.join(root, "server_results", "ls_noise30.log")])
    else:
        folds = pd.concat([pd.read_csv(os.path.join(root, "server_results", w, "results",
                                                    "longest_stable_folds.csv"))
                           for w in ("longest_stable_run", "longest_stable_run_sh")])
        walls = log_walltimes(
            sorted(glob.glob(os.path.join(root, "server_results", "longest_stable_run",
                                          "logs", "*.log")))
            + sorted(glob.glob(os.path.join(root, "server_results", "longest_stable_run_sh",
                                            "**", "*.log"), recursive=True)))
    n_methods = folds["Method"].nunique()
    for (model, dataset), group in folds.groupby(["Model", "Dataset"]):
        # Derive_Seconds repeats across the methods of a fold; count each fold once.
        derive = group.drop_duplicates(["Repeat", "Fold"])["Derive_Seconds"].sum()
        wall = walls.get((model, dataset))
        scoring = max(0.0, (wall - derive) / n_methods) if wall is not None else np.nan
        for method, name in (("Window_w3", "WAda3"), ("LongestStable", "LSW")):
            part = group[group["Method"] == method]
            if len(part):
                add((model, dataset, name), part["Accuracy"].mean(), derive + scoring)

    # WAda with w = 10. The longest-run study scored only w = 2 and w = 3, so the
    # accuracy comes from the window sweep, which is the run that covers w = 10 on
    # all fourteen datasets. The cost is that of w = 3: every window is read off
    # the same correctness matrix, so the window rules and LSW are charged alike.
    sweep = os.path.join(root, "server_results",
                         "sens_noise30" if noisy else "sens1", "results", "figures")
    for model, _, _ in RULES:
        f = os.path.join(sweep, f"w_sweep_{model}_summary.csv")
        if not os.path.exists(f):
            continue
        for _, r in pd.read_csv(f).query("w == 10").iterrows():
            base = sec.get((model, r["Dataset"], "WAda3"))
            if base is not None:
                add((model, r["Dataset"], "WAda10"), r["Accuracy"], base)
    return acc, sec


def summarise(acc, sec):
    rows = []
    for model, pretty, colour in RULES:
        for method in MARK:
            a = [acc[(model, d, method)] for d in DS if (model, d, method) in acc]
            s = [sec[(model, d, method)] for d in DS if (model, d, method) in sec]
            if len(a) == len(DS) and not np.isnan(np.sum(s)):
                rows.append({"Rule": pretty, "Method": method,
                             "Name": re.sub(r"[${}]", "", label(pretty, method)),
                             "Colour": colour,
                             "Accuracy": np.mean(a) * 100, "Seconds": np.mean(s)})
    return pd.DataFrame(rows)


def skyline(frame):
    """Non-dominated set: nothing is both more accurate and faster."""
    keep = []
    for _, r in frame.iterrows():
        if not ((frame["Accuracy"] >= r["Accuracy"]) & (frame["Seconds"] <= r["Seconds"]) &
                ((frame["Accuracy"] > r["Accuracy"]) | (frame["Seconds"] < r["Seconds"]))).any():
            keep.append(r)
    return pd.DataFrame(keep).sort_values("Seconds")


def draw(frame, title, out_png, tag):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    front = skyline(frame)
    if len(front) > 1:
        ax.step(front["Seconds"], front["Accuracy"], where="post",
                color="#999", lw=1.2, ls="--", zorder=1, label="skyline")

    for _, r in frame.iterrows():
        ax.scatter(r["Seconds"], r["Accuracy"], marker=MARK[r["Method"]], s=95,
                   color=r["Colour"], edgecolors="white", linewidths=1.0, zorder=3)
        on_front = ((front["Rule"] == r["Rule"]) & (front["Method"] == r["Method"])).any()
        dx, dy = NUDGE[tag].get((r["Rule"], r["Method"]), (7, 4))
        ax.annotate(label(r["Rule"], r["Method"]), (r["Seconds"], r["Accuracy"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=7.0,
                    ha="right" if dx < 0 else "left",
                    color=r["Colour"], fontweight="bold" if on_front else "normal")

    ax.set_xscale("log")
    # room for the labels that hang off the rightmost points
    lo, hi = frame["Seconds"].min(), frame["Seconds"].max()
    ax.set_xlim(lo * 0.55, hi * 2.6)
    ax.set_xlabel("average GPU time per dataset (s, log scale)")
    ax.set_ylabel("average accuracy (%)")
    ax.set_title(title, loc="left", fontsize=12)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=p) for _, p, c in RULES]
    handles += [plt.Line2D([], [], marker=MARK[m], ls="", color="#666", label=SHORT[m])
                for m in MARK]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, facecolor=BG)
    plt.close(fig)
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for noisy, tag, title in ((False, "clean", "Accuracy against cost, clean data"),
                              (True, "noise30", "Accuracy against cost, 30% label noise")):
        acc, sec = gather(args.root, noisy)
        frame = summarise(acc, sec)
        png = os.path.join(args.out, f"skyline_{tag}.png")
        front = draw(frame, title, png, tag)
        frame.to_csv(os.path.join(args.out, f"skyline_{tag}.csv"), index=False)
        print(f"\n=== {tag} ===  {len(frame)} points -> {png}")
        print(frame.sort_values("Seconds")[["Rule", "Name", "Accuracy", "Seconds"]]
              .to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
        print("skyline: " + ", ".join(f"{r['Name']}" for _, r in front.iterrows()))


if __name__ == "__main__":
    main()
