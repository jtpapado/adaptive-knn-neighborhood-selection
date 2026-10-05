# Dynamic Selection of Neighbourhood Size in Standard, Mutual and Symmetric k-Nearest Neighbour Classification

Code and data for the paper by Ioannis Papadopoulos, Stefanos Ougiaroglou, Katerina Asdre
and Leonidas Akritidis (Department of Information and Electronic Engineering,
International Hellenic University).

The paper extends the adaptive neighbourhood-size selection of AdaNN to the Mutual and
Symmetric k-NN rules (AdaMNN, AdaSNN), adds window-based variants (WAdaNN, WAdaMNN,
WAdaSNN) and introduces a window-parameter-free **longest-stable** criterion (LAdaNN,
LAdaMNN, LAdaSNN, referred to as LSW). Everything is evaluated on 14 benchmark datasets
by stratified 10-fold cross-validation, on the original data and under 30% artificially
introduced label noise.

This repository contains only the code behind the reported results.

---

## Contents

| File | Role |
|---|---|
| `knn_algorithms.py` | The classifiers: directed, Mutual and Symmetric k-NN, with the adaptive, window-based and longest-stable neighbourhood-size rules. CPU (NumPy) and GPU (PyTorch) backends. |
| `knn_pipeline.py` | The main experiment pipeline: cross-validation design, nested tuning, label-noise injection, timing, Friedman/Bonferroni–Dunn and Wilcoxon tests. |
| `longest_stable_k.py` | The longest-stable criterion and the ablation over its three read-offs (start, midpoint, endpoint), with paired Nadeau–Bengio corrected resampled *t*-tests. |
| `sensitivity_study.py` | The two sensitivity analyses: `k_max` over 1–50 and the window `w` over 2–15. |
| `skyline.py` | The accuracy–efficiency skyline diagrams (Figures 5 and 6), drawn from the results of the runs above. |
| `run_experiments.sh` | Driver: the full experiment set on the original data. |
| `run_experiments_noise.sh` | Driver: the same set at 30% label noise. |
| `run_longest_stable.sh` | Driver: the longest-stable criterion and its baselines. |
| `datasets/` | The 14 datasets, unscaled. |
| `reference_configs/` | The `config.yaml` files the drivers wrote for the runs reported in the paper, included so the exact settings can be inspected without running anything. |

---

## Requirements

Python 3.10.12 with the packages in `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch is needed only for the GPU backend. Every script also accepts
`--backend cpu` and then runs on NumPy alone, considerably more slowly.

The reported experiments were run on a server with an NVIDIA A16 GPU. All execution
times in the paper are GPU times measured with CUDA events around the device work, so
absolute timings will differ on other hardware; the relative comparison between methods
is what the paper draws on.

---

## Datasets

The 14 datasets come from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml)
and the [KEEL repository](https://sci2s.ugr.es/keel/). They are included here in the layout
the code expects:

```
datasets/<name>/data1      # first portion
datasets/<name>/data_te1   # second portion
```

Each file is tab-separated with no header; the last column is an integer class label and
every other column is a numeric feature. The two files are **pooled** and the pooled set
is then partitioned by stratified 10-fold cross-validation, so the original split plays no
part in the evaluation.

The values are unscaled. Min-max normalisation is fitted inside each training fold and
applied unchanged to the held-out fold, so the held-out rows never influence the
transformation applied to the training rows.

Dataset keys: `mgt` `pd` `ls` `ph` `bn` `ys` `pm` `txr` `sh` `ecl` `iris` `tn` `bl` `lir`
(Magic Gamma Telescope, Pen-Digits, Landsat Satellite, Phoneme, Banana, Yeast, Pima Indians
Diabetes, Texture, Shuttle, E. coli, Iris, Twonorm, Balance Scale, Letter Recognition).

---

## Reproducing the experiments

All four entry points are **resumable**: each (dataset, method) result is written as soon
as it completes and skipped on a re-run, so interrupting costs only the cell in flight.
Each run writes into its own workspace directory, which keeps its config, logs and results
together.

`skyline.py` reads the finished runs by name, so the workspace names below are the ones it
expects, and it expects them to sit together under a directory called `server_results/`:

```
server_results/srv_gpu/  srv_gpu_sh/  srv_noise30b/
                longest_stable_run/  longest_stable_run_sh/  ls_noise30/
                sens1/  sens_noise30/
```

Only the figure step cares about these names; every other script is indifferent to them.

### 1. Original data

```bash
./run_experiments.sh --workspace server_results/srv_gpu --datasets "iris ecl bl pm ys bn ph txr ls tn pd mgt lir"
./run_experiments.sh --workspace server_results/srv_gpu_sh --datasets "sh"
```

Shuttle is run into its own workspace because it costs roughly as much as the other
thirteen together.

Produces, for the Directed, Mutual and Symmetric rules: fixed *k* = 5, nested Best *k*,
nested tuned-window WAda<sub>w*</sub>, and AdaNN with `k_max` = 9 and 30, together with
execution times and the statistical tests.

### 2. 30% label noise

```bash
./run_experiments_noise.sh --workspace server_results/srv_noise30b
```

The same set with 30% of each fold's **training** labels replaced by a randomly chosen
alternative class. Held-out labels are never touched. The corruption is seeded on
(seed, dataset, repeat, fold), so the noisy training sets match those used by the
sensitivity and longest-stable studies at the same rate.

### 3. The longest-stable criterion

```bash
./run_longest_stable.sh --workspace server_results/longest_stable_run --windows "2 3"
python3 longest_stable_k.py --workspace server_results/ls_noise30 --noise 0.3 \
        --windows 2 3 --backend gpu 2>&1 | tee server_results/ls_noise30.log
```

Scores the three read-offs of the longest stable run against AdaNN and the fixed window
over identical folds. Shuttle (`sh`) is run last and into its own workspace because it is
roughly as expensive as the other thirteen together; `--skip-sh` and `--only-sh` control
this.

### 4. Sensitivity analyses

```bash
python3 sensitivity_study.py --workspace server_results/sens1         --backend gpu
python3 sensitivity_study.py --workspace server_results/sens_noise30  --backend gpu --noise 0.3
```

Sweeps `k_max` over 1–50 and `w` over 2–15 for all three rules on all 14 datasets, and
writes the per-window summaries the paper draws on, including the mean selected *k* and
the fraction of training instances that fall back to `k_max`.

Add `--figures-only` to redraw the figures from results already on disk.

### 5. The skyline diagrams

Once the four runs above have finished:

```bash
python3 skyline.py --root . --out figures
```

Writes `figures/skyline_clean.png` and `figures/skyline_noise30.png` (Figures 5 and 6),
each with the underlying points as a `.csv`, and prints the non-dominated set for each
condition.
