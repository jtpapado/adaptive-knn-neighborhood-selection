import os
import re
import sys
import time
import math
import zlib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from knn_algorithms import CustomKNN, MutualKNN, SymmetricKNN, set_backend, get_backend, set_gpu_max_elements
import multiprocessing as mp
import shutil
import numpy as np
from collections import Counter, namedtuple
from scipy import stats
from itertools import combinations
import yaml

# ============================================================================
# WORKSPACE
#   Everything a run generates lives in one folder: its config.yaml, the
#   results, the adaptive datasets, the noisy copies.  Naming another
#   workspace gives a clean slate with its own config:
#
#       python knn_pipeline.py                        # workspace1 (default)
#       python knn_pipeline.py --workspace workspace2 # starts from scratch
#
#   Only the original datasets and these scripts live outside a workspace.
# ============================================================================
DEFAULT_WORKSPACE = "workspace1"

def _resolve_workspace():
    for i, arg in enumerate(sys.argv):
        if arg == "--workspace" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--workspace="):
            return arg.split("=", 1)[1]
    return os.environ.get("KNN_WORKSPACE") or DEFAULT_WORKSPACE

WORKSPACE_DIR = _resolve_workspace()
# Spawned workers re-import this module with a different argv, so the choice
# has to travel in the environment for them to read the same config.
os.environ["KNN_WORKSPACE"] = WORKSPACE_DIR

def workspace_file(*parts):
    """Path inside the active workspace."""
    return os.path.join(WORKSPACE_DIR, *parts)


# ============================================================================
# CONFIGURATION  —  loaded from <workspace>/config.yaml
# ============================================================================
_CONFIG_FILE = workspace_file("config.yaml")

def _legacy_artifacts():
    """Generated files/folders sitting next to the scripts from before workspaces."""
    found = []
    for name in sorted(os.listdir(".")):
        if name in ("config.yaml", "config.yaml.bak", "results"):
            found.append(name)
        elif os.path.isdir(name) and (
                name.startswith("adaptive_")
                or re.fullmatch(r"noise[\dp]+_seed\d+", name)
                or re.search(r"_noise[\dp]+_seed\d+$", name)):
            found.append(name)
        elif os.path.isfile(name) and name.startswith("results_phase"):
            found.append(name)
    return found

def _adopt_legacy_layout():
    """Move a pre-workspace layout into the default workspace, once."""
    artifacts = _legacy_artifacts()
    if not artifacts:
        return False
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    print(f"📦 Moving the existing layout into {WORKSPACE_DIR}/ "
          f"({len(artifacts)} item(s)) — the datasets and scripts stay where they are.")
    for name in artifacts:
        target = workspace_file(name)
        if name.startswith("results_phase") and os.path.isfile(name):
            os.makedirs(workspace_file("results"), exist_ok=True)
            target = workspace_file("results", name)
        if os.path.exists(target):
            continue
        try:
            shutil.move(name, target)
            print(f"   {name} → {target}")
        except OSError as e:
            print(f"   ⚠ could not move {name}: {e}")
    return os.path.exists(_CONFIG_FILE)

def _seed_workspace_config(path):
    """A new workspace starts from the default workspace's config, if there is one."""
    template = os.path.join(DEFAULT_WORKSPACE, "config.yaml")
    if os.path.abspath(template) == os.path.abspath(path) or not os.path.exists(template):
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    shutil.copy2(template, path)
    print(f"📋 New workspace: copied {template} → {path}")
    print("   Edit it to change the settings for this workspace.")
    return True

def _load_config(path=_CONFIG_FILE):
    if not os.path.exists(path):
        # First run after the workspace layout was introduced?
        if os.path.abspath(WORKSPACE_DIR) == os.path.abspath(DEFAULT_WORKSPACE):
            _adopt_legacy_layout()
        if not os.path.exists(path):
            _seed_workspace_config(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Configuration file '{path}' not found. Create it (or copy one from "
            f"another workspace) before running the pipeline."
        )
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

_CFG = _load_config()

# Backend: "cpu" (NumPy) or "gpu" (PyTorch with CUDA/MPS/CPU-fallback)
_BACKEND_CFG = _CFG.get("backend", "cpu")
set_backend(_BACKEND_CFG)

# GPU batch size — controls max float32 elements per distance-matrix batch on GPU.
# Lower this if you hit CUDA OOM on large datasets.
_GPU_MAX_ELEM = _CFG.get("gpu_max_elements", 32_000_000)
set_gpu_max_elements(_GPU_MAX_ELEM)

# Timing: "cpu" (process_time) or "gpu" (CUDA events / perf_counter fallback)
_TIMING_MODE = _CFG.get("timing", "cpu").lower()


class Timer:
    """
    Unified timer that respects the configured timing mode.

      "cpu"  — time.process_time()  (CPU-only, no I/O or GPU waits)
      "gpu"  — torch.cuda.Event    (GPU-synchronised wall-clock)
               Falls back to time.perf_counter() on MPS / CPU-only PyTorch.

    Usage:
        t = Timer()
        t.start()
        ...  # work
        elapsed = t.stop()   # seconds (float)
    """

    def __init__(self):
        self._mode = _TIMING_MODE
        self._use_cuda_events = False
        self._start_event = None
        self._end_event = None
        self._start_time = None

        if self._mode == "gpu":
            try:
                import torch
                if torch.cuda.is_available():
                    self._use_cuda_events = True
            except ImportError:
                pass

    def start(self):
        if self._use_cuda_events:
            import torch
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        elif self._mode == "gpu":
            # GPU backend but no CUDA (MPS or CPU-fallback) → wall-clock
            self._start_time = time.perf_counter()
        else:
            self._start_time = time.process_time()

    def stop(self) -> float:
        """Return elapsed time in seconds."""
        if self._use_cuda_events:
            import torch
            self._end_event.record()
            torch.cuda.synchronize()
            return self._start_event.elapsed_time(self._end_event) / 1000.0  # ms → s
        elif self._mode == "gpu":
            return time.perf_counter() - self._start_time
        else:
            return time.process_time() - self._start_time

    @staticmethod
    def label() -> str:
        """Human-readable label for output headers."""
        if _TIMING_MODE == "gpu":
            return "GPU Time (s)"
        return "CPU Time (s)"

BASE_FOLDER          = _CFG["base_folder"]
INITIAL_K_VALUES     = _CFG["phase1"]["k_values"]
CV_K_RANGE           = range(_CFG["phase2"]["k_min"], _CFG["phase2"]["k_max"] + 1)
CV_FOLDS             = _CFG["phase2"]["cv_folds"]

# ----------------------------------------------------------------------------
# CROSS-VALIDATION — the evaluation protocol for every phase.
#   folds/repeats  — how the pooled data1+data_te1 is cut for scoring.
#   bestk_*        — the grid search in Phase 2 gets its own, heavier setting:
#                    the k it picks feeds Phase 3 and the reported tables, so it
#                    is worth averaging over more partitions than a single score.
#   scaling        — "minmax" fits the scaler on each fold's TRAINING rows.
#                    Point base_folder at an UNSCALED copy of the data for
#                    this to mean anything; datasets/ holds it unscaled.
# ----------------------------------------------------------------------------
_CV_CFG                = _CFG.get("cv") or {}
CV_EVAL_FOLDS          = int(_CV_CFG.get("folds", 10))
CV_EVAL_REPEATS        = int(_CV_CFG.get("repeats", 1))
CV_BESTK_FOLDS         = int(_CV_CFG.get("bestk_folds", 10))
CV_BESTK_REPEATS       = int(_CV_CFG.get("bestk_repeats", 10))
CV_SEED                = int(_CV_CFG.get("seed", 42))
CV_SCALING             = _CV_CFG.get("scaling", "minmax")
CV_RARE_CLASS_POLICY   = _CV_CFG.get("rare_class_policy", "warn")
CV_TIE_BREAKING      = _CFG["phase2"].get("tie_breaking", "lowest_k")

# Phase 4/14 — nested (double) cross-validation. Optional "phase4:" key;
# the defaults are the protocol reviewers usually ask for (10 x 10-fold).
_NESTED_CFG                = _CFG.get("phase4") or {}
NESTED_OUTER_FOLDS         = int(_NESTED_CFG.get("outer_folds", 10))
NESTED_INNER_FOLDS         = int(_NESTED_CFG.get("inner_folds", 5))
NESTED_REPEATS             = int(_NESTED_CFG.get("repeats", 10))
NESTED_SEED                = int(_NESTED_CFG.get("seed", 42))
# What to do with classes too small to appear in every fold (ecoli has
# classes with 2 members): "warn" | "reduce" | "drop"  — see stratified_plan().
NESTED_RARE_CLASS_POLICY   = _NESTED_CFG.get("rare_class_policy", "warn")
# Which method families to score: bestk | adaptive | sqrt | lastk
# Default scope: the three neighbourhood rules (Custom/Mutual/Symmetric kNN)
# with k tuned inside every outer fold. The adaptive/sqrt/lastk families can be
# switched on, but each one re-derives its per-row k in every fold.
NESTED_METHODS             = _NESTED_CFG.get("methods") or ["bestk"]
# Candidate windows for the "lastk_tuned" family, which picks w inside every
# outer fold instead of fixing it. Sweeping w showed no value that is safe
# across datasets, so quoting a fixed-w result understates the method; this is
# the honest way to report it.
NESTED_W_VALUES            = [int(w) for w in
                              (_NESTED_CFG.get("w_values") or range(2, 16))]
# Fixed neighbourhood sizes scored under the same outer folds as everything
# else, so the adaptive families have an untuned baseline to be measured against.
NESTED_FIXED_K             = [int(k) for k in (_NESTED_CFG.get("fixed_k") or [5])]
ADAPTIVE_K_MAX_VALUES = _CFG["phase5"]["k_max_values"]
LAST_K_MAX           = _CFG["phase9"]["k_max"]
# The stability window is swept, the way phase5.k_max_values is. A scalar
# "stability_window:" is still accepted — that is what the GUI writes — and is
# read as a one-value sweep.
_WINDOWS_CFG         = _CFG["phase9"].get("stability_windows")
if _WINDOWS_CFG is None:
    _WINDOWS_CFG     = [_CFG["phase9"].get("stability_window", 3)]
elif not isinstance(_WINDOWS_CFG, (list, tuple)):
    _WINDOWS_CFG     = [_WINDOWS_CFG]
STABILITY_WINDOWS    = [int(w) for w in _WINDOWS_CFG]
# Kept for the places that still want a single value (banners, legacy CSVs).
STABILITY_WINDOW     = STABILITY_WINDOWS[0]

# Worker counts are named after what a phase DOES, not after its number: the
# phases were renumbered once already, and keys like "phases_5_7_9" silently
# changed meaning across that rename. Old keys are still read, mapped to the
# role they had when they were written.
_W = _CFG["workers"]

def _workers(role, legacy, default):
    value = _W.get(role)
    if value is None:
        value = _W.get(legacy, default)
    return value

_raw_workers          = _workers("main", "phases_1_to_3", -1)
MAX_WORKERS           = max(1, mp.cpu_count() - 1) if _raw_workers == -1 else max(1, _raw_workers)
# Phases 5/7/9 build the derived datasets — RAM intensive, usually sequential.
BUILDER_WORKERS       = max(1, _workers("builders", "phases_4_6_8", 1))
# Phases 6/8/10 score them.
TESTER_WORKERS        = max(1, _workers("testers", "phases_5_7_9", 10))

# GPU backend → force sequential execution.
# Each spawned process copies the full PyTorch + CUDA runtime (~1-2 GB),
# so multiple workers quickly exhaust RAM (especially on WSL).
# The GPU already parallelises internally, so multiprocessing adds no speed.
if _BACKEND_CFG == "gpu":
    MAX_WORKERS     = 1
    BUILDER_WORKERS = 1
    TESTER_WORKERS  = 1
    _MP_CTX = mp.get_context("spawn")
else:
    _MP_CTX = mp.get_context()  # platform default ('fork' on Linux)

# Phase toggles — default True so omitting a key never silently disables a phase
_PHASES               = _CFG.get("phases", {})
N_PHASES              = 13
ENABLED_PHASES        = {f"phase{i}": _PHASES.get(f"phase{i}", True)
                         for i in range(1, N_PHASES + 1)}
# Phase 4 (nested CV) defaults to OFF: it re-tunes k inside every outer fold, so
# a full run costs roughly outer_folds x repeats times a single Phase 2+3 pass.
# Launch it deliberately — "python knn_pipeline.py --nested-cv" — or switch it
# on in the phases: block of config.yaml.
ENABLED_PHASES["phase4"] = _PHASES.get("phase4", False)

# Models to evaluate — optional "models:" key in config.yaml.
# Omitting the key (or leaving it empty) keeps all three models.
ALL_MODELS            = ['CustomKNN', 'MutualKNN', 'SymmetricKNN']
_cfg_models           = _CFG.get("models") or ALL_MODELS
MODELS                = [m for m in ALL_MODELS if m in _cfg_models] or list(ALL_MODELS)

# Dataset selection — optional "datasets:" key in config.yaml:
#   datasets:
#     include: [iris, bl]   # empty/absent = every sub-folder of base_folder
#     exclude: [ys]
_DATASETS_CFG         = _CFG.get("datasets") or {}

# Where every result file is written — always inside the workspace.
OUTPUT_FOLDER         = workspace_file(_CFG.get("output_folder") or "results")
# One extra workbook per dataset, written at the end of a run. Optional key.
PER_DATASET_RESULTS   = _CFG.get("per_dataset_results", True)

# ----------------------------------------------------------------------------
# LABEL NOISE — optional "noise:" key in config.yaml
#   enabled   — build noisy copies of every dataset and run all phases on them
#   rates     — one or more fractions of TRAINING labels to flip; each becomes
#               its own dataset copy and its own full run.
#               (0.30 or 30 both mean 30 %.  "rate: 0.3" is still accepted.)
#   seed      — makes the noisy copies exactly reproducible
#   run_clean — also run every phase on the untouched data (default true)
# The clean datasets are never modified: each noisy copy is a new folder named
# <base_folder>_noise<pct>_seed<seed>, and its results/derived datasets live in
# their own sub-folders so nothing clean is overwritten.
# ----------------------------------------------------------------------------
_NOISE_CFG            = _CFG.get("noise") or {}
NOISE_ENABLED         = bool(_NOISE_CFG.get("enabled", False))

def _normalise_rates(raw):
    """Accept 0.3 / 30 / [10, 20, 30] and return sorted unique fractions."""
    if raw is None:
        raw = 0.30
    if not isinstance(raw, (list, tuple, set)):
        raw = [raw]
    rates = []
    for value in raw:
        value = float(value)
        rates.append(value / 100.0 if value > 1 else value)
    return sorted(set(rates))

NOISE_RATES           = _normalise_rates(_NOISE_CFG.get("rates", _NOISE_CFG.get("rate")))
NOISE_SEED            = int(_NOISE_CFG.get("seed", 42))
NOISE_RUN_CLEAN       = bool(_NOISE_CFG.get("run_clean", True))

CLEAN_BASE_FOLDER     = BASE_FOLDER          # outside the workspace, never written to
# Noise rate of the variant currently being run (None = clean). Phases 1-13 read
# the pre-built noisy files; Phase 4 re-injects noise per fold and needs the rate.
ACTIVE_NOISE_RATE     = None

# --- active variant (switched by activate_variant() while main() runs) -------
VARIANT_DIR           = ""                   # "" = clean, else the noise tag
RESULTS_FOLDER        = OUTPUT_FOLDER
PER_DATASET_FOLDER    = os.path.join(RESULTS_FOLDER, "per_dataset")

# ============================================================================
# UTILITIES
# ============================================================================

def results_path(filename):
    """Full path of a result file inside the results folder (created on demand)."""
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    return os.path.join(RESULTS_FOLDER, filename)

def derived_path(name):
    """
    Folder for a generated dataset collection, inside the workspace.
    The clean variant sits at the top of it; each noisy variant gets a sub-folder.
    """
    return workspace_file(VARIANT_DIR, name) if VARIANT_DIR else workspace_file(name)

def adaptive_folder(k_max):
    return derived_path(f"adaptive_k{k_max}")

def sqrt_folder():
    return derived_path("adaptive_sqrt_n2")

def last_k_folder(window=None):
    window = STABILITY_WINDOW if window is None else window
    return derived_path(f"adaptive_last_k{LAST_K_MAX}_w{window}")

def merge_timing_csv(prefix, new_records, key_columns):
    """
    Write the timing CSV for phases 4/6/8, carrying earlier rows forward.

    These phases only build the datasets that are still missing, so a resumed
    run produces records for those alone. Without merging, the newest CSV — the
    only one anything reads later — would hold just that remainder and the rows
    from the interrupted run would be lost.
    """
    table = pd.DataFrame(new_records)
    previous_path = _get_latest_csv(prefix)
    if previous_path and os.path.exists(previous_path):
        try:
            previous = pd.read_csv(previous_path)
            keys = [c for c in key_columns
                    if c in previous.columns and c in table.columns]
            if keys and len(previous):
                fresh = set(map(tuple, table[keys].astype(str).to_numpy()))
                keep = [tuple(row) not in fresh
                        for row in previous[keys].astype(str).to_numpy()]
                previous = previous[keep]
            table = pd.concat([previous, table], ignore_index=True)
        except Exception as e:
            print(f"⚠ Could not merge earlier timings from {previous_path}: {e}")
    sort_by = [c for c in key_columns if c in table.columns]
    if sort_by:
        table = table.sort_values(sort_by).reset_index(drop=True)
    timestamp = datetime.now().strftime("%y_%m_%d_%H%M%S")
    out_file = results_path(f"{prefix}{timestamp}.csv")
    table.to_csv(out_file, index=False)
    return out_file

def migrate_legacy_results():
    """
    Move result files left in the working directory by older runs into the
    results folder, so resuming still finds them.

    Only ever runs for the clean variant — a stray file predates the noise
    feature, so it can only belong to the clean run.
    """
    if VARIANT_DIR:
        return []
    stray = []
    for folder in (".", WORKSPACE_DIR):
        if not os.path.isdir(folder):
            continue
        stray += [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                  if f.startswith('results_phase')
                  and (f.endswith('.xlsx') or f.endswith('.csv'))
                  and os.path.isfile(os.path.join(folder, f))]
    moved = []
    for path in stray:
        target = results_path(os.path.basename(path))
        if os.path.exists(target) or os.path.abspath(path) == os.path.abspath(target):
            continue
        try:
            shutil.move(path, target)
            moved.append(path)
        except OSError as e:
            print(f"⚠ Could not move {path} into {RESULTS_FOLDER}/: {e}")
    if moved:
        print(f"📦 Moved {len(moved)} earlier result file(s) into {RESULTS_FOLDER}/")
    return moved

def list_datasets(base_folder):
    """Dataset sub-folders of base_folder, honouring the optional include/exclude config."""
    folders = sorted([f for f in os.listdir(base_folder)
                      if os.path.isdir(os.path.join(base_folder, f))])
    include = _DATASETS_CFG.get("include") or None
    exclude = set(_DATASETS_CFG.get("exclude") or [])
    if include:
        include = set(include)
        folders = [f for f in folders if f in include]
    return [f for f in folders if f not in exclude]

def load_dataset(file_path):
    data = pd.read_csv(file_path, sep="\t", header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def load_dataset_with_df(file_path):
    data = pd.read_csv(file_path, sep="\t", header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y, data

def get_latest_file(pattern):
    if not os.path.isdir(RESULTS_FOLDER):
        return None
    files = [f for f in os.listdir(RESULTS_FOLDER)
             if f.startswith(pattern)
             and f.endswith('.xlsx')
             and not f.endswith('_temp.xlsx')]
    if files:
        files.sort(reverse=True)
        return os.path.join(RESULTS_FOLDER, files[0])
    return None

def noise_tag(rate, seed=None):
    """Folder/suffix tag for a noise setting, e.g. 30 % + seed 42 → noise30_seed42."""
    seed = NOISE_SEED if seed is None else seed
    percent = f"{rate * 100:g}".replace(".", "p")     # 12.5 → "12p5"
    return f"noise{percent}_seed{int(seed)}"

def noisy_base_folder(rate, base_folder=None, seed=None):
    """The noisy copy is generated, so it lives inside the workspace."""
    base = CLEAN_BASE_FOLDER if base_folder is None else base_folder
    name = os.path.basename(os.path.normpath(base))
    return workspace_file(f"{name}_{noise_tag(rate, seed)}")

def _dataset_rng(dataset_name, seed):
    """
    A separate random stream per dataset, derived from the run seed.

    Deriving it from the dataset NAME (not from the loop order) means a dataset
    always gets the same flips for a given seed, no matter which datasets are
    selected or in what order they are processed.
    """
    return np.random.default_rng([int(seed), zlib.crc32(dataset_name.encode("utf-8"))])

def inject_label_noise(y, rate, rng):
    """
    Flip each label with probability `rate`, uniformly to one of the OTHER
    classes. Returns (noisy labels, number flipped).
    """
    classes, codes = np.unique(y, return_inverse=True)
    if len(classes) < 2:
        return np.array(y, copy=True), 0
    flip = rng.random(len(y)) <= rate
    n_flipped = int(flip.sum())
    if n_flipped:
        codes = codes.copy()
        # +1..C-1 modulo C lands on any class except the original, uniformly
        offset = rng.integers(1, len(classes), size=n_flipped)
        codes[flip] = (codes[flip] + offset) % len(classes)
    return classes[codes], n_flipped

def create_all_noisy_datasets(rates=None, base_folder=None, seed=None, datasets=None):
    """Build one noisy copy of the collection per noise rate."""
    rates = NOISE_RATES if rates is None else _normalise_rates(rates)
    return [create_noisy_datasets(rate, base_folder, seed, datasets) for rate in rates]


def create_noisy_datasets(rate, base_folder=None, seed=None, datasets=None):
    """
    Write <base_folder>_noise<pct>_seed<seed>/<dataset>/{data1,data_te1}.

    Existing noisy datasets are kept as they are, so the step is resumable and
    re-running it never changes results that were already produced from it.
    """
    base_folder = CLEAN_BASE_FOLDER if base_folder is None else base_folder
    seed = NOISE_SEED if seed is None else seed
    out_base = noisy_base_folder(rate, base_folder, seed)

    print("\n" + "="*80)
    print(f"LABEL NOISE: {rate*100:g}% of training labels, seed {seed}")
    print("="*80 + "\n")

    if os.path.abspath(out_base) == os.path.abspath(base_folder):
        raise ValueError("The noisy folder would overwrite the clean one — aborting.")
    if not os.path.isdir(base_folder):
        print(f"✗ Base folder '{base_folder}' not found.\n")
        return None
    if not 0.0 < rate < 1.0:
        print(f"✗ Noise rate must be between 0 and 1 (got {rate}).\n")
        return None

    names = datasets if datasets is not None else list_datasets(base_folder)
    print(f"Source: {base_folder}/  ({len(names)} dataset(s))")
    print(f"Target: {out_base}/\n")
    os.makedirs(out_base, exist_ok=True)

    records = []
    for dataset_name in names:
        in_folder = os.path.join(base_folder, dataset_name)
        out_folder = os.path.join(out_base, dataset_name)
        train_in = os.path.join(in_folder, "data1")
        train_out = os.path.join(out_folder, "data1")
        if not os.path.exists(train_in):
            print(f"⚠ Skipping {dataset_name} (missing data1)")
            continue
        if os.path.exists(train_out):
            print(f"⊘ Skipping (exists): {dataset_name}")
            continue

        os.makedirs(out_folder, exist_ok=True)
        _, y, df_train = load_dataset_with_df(train_in)
        y_noisy, n_flipped = inject_label_noise(y, rate, _dataset_rng(dataset_name, seed))

        df_noisy = df_train.copy()
        df_noisy[df_noisy.columns[-1]] = y_noisy
        df_noisy.to_csv(train_out, sep="\t", header=False, index=False)

        test_in = os.path.join(in_folder, "data_te1")
        if os.path.exists(test_in):
            shutil.copy2(test_in, os.path.join(out_folder, "data_te1"))

        actual = n_flipped / len(y) if len(y) else 0.0
        print(f"✓ {dataset_name:20s} flipped {n_flipped}/{len(y)} labels ({actual*100:.1f}%)")
        records.append({"Dataset": dataset_name, "Train_Rows": len(y),
                        "Flipped": n_flipped, "Actual_Rate": round(actual, 6),
                        "Requested_Rate": rate, "Seed": int(seed),
                        "Classes": int(len(np.unique(y)))})

    manifest = os.path.join(out_base, "noise_manifest.csv")
    if records:
        table = pd.DataFrame(records)
        if os.path.exists(manifest):
            try:
                previous = pd.read_csv(manifest)
                previous = previous[~previous["Dataset"].isin(table["Dataset"])]
                table = pd.concat([previous, table], ignore_index=True)
            except Exception:
                pass
        table.sort_values("Dataset").to_csv(manifest, index=False)
        print(f"\n📝 Noise manifest: {manifest}")
    print(f"\n✓ Noisy datasets ready in {out_base}/\n")
    return out_base


# ============================================================================
# CROSS-VALIDATION HARNESS
#
#   Every phase scores its methods the same way: pool data1 and data_te1, cut
#   them with stratified k-fold, and report mean +- std over the folds.  The
#   fixed 70/30 split the earlier protocol used is gone — on datasets this size
#   one split carried several points of sampling noise, and the statistical
#   tests downstream inherited all of it.
#
#   Two rules hold everywhere, and both exist to keep the test rows unseen:
#
#     * The min-max scaler is fitted on the TRAINING rows of each fold.  The
#       Scaling once over the whole dataset would let each column's extremes
#       leak from the held-out rows into the training rows; datasets/ holds
#       the unscaled values so the scaling can happen here instead.  Test values
#       are expected to fall outside [0, 1] — that is the sign it worked.
#     * Label noise, where a run uses it, is injected into the training rows of
#       each fold and never into the test rows.
# ============================================================================

def cv_design(X, y, folds, repeats, seed, policy=None):
    """
    The splits one dataset gets, after dropping whatever stratification cannot
    honour.  Returns (X, y, splits, n_splits, note) or None when the dataset
    cannot be stratified at all; each split is (repeat, fold, train_idx, test_idx).
    """
    policy = CV_RARE_CLASS_POLICY if policy is None else policy
    n_splits, keep, note = stratified_plan(y, folds, policy)
    if n_splits < 2:
        return None
    X, y = X[keep], y[keep]
    if repeats > 1:
        splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats,
                                           random_state=seed)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = [(i // n_splits + 1, i % n_splits + 1, tr, te)
              for i, (tr, te) in enumerate(splitter.split(X, y))]
    return X, y, splits, n_splits, note


def scale_fold(X_train, X_test):
    """min-max fitted on the training rows alone. Test values may leave [0, 1]."""
    if CV_SCALING != "minmax":
        return X_train, X_test
    lo = X_train.min(axis=0)
    span = X_train.max(axis=0) - lo
    span = np.where(span == 0, 1.0, span)
    return (X_train - lo) / span, (X_test - lo) / span


def fold_labels(y_train, dataset_name, repeat, fold, noise_rate):
    """Training labels for one fold, corrupted if the run is a noisy one."""
    if not noise_rate:
        return y_train, 0
    rng = np.random.default_rng(
        [int(NOISE_SEED), zlib.crc32(dataset_name.encode("utf-8")), repeat, fold])
    return inject_label_noise(y_train, noise_rate, rng)


def fold_metrics(y_true, y_pred, elapsed):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        Timer.label(): elapsed,
    }


_CV_METRICS = ["Accuracy", "Precision", "Recall", "F1"]


def summarise_cv(folds_df, keys):
    """mean +- std over the folds, plus the ready-to-paste "m ± s" columns."""
    tcol = _fold_time_column(folds_df) or Timer.label()
    spec = {"Folds": ("Accuracy", "size")}
    for metric in _CV_METRICS:
        spec[f"{metric}_Mean"] = (metric, "mean")
        spec[f"{metric}_Std"] = (metric, "std")
    spec["Time_Mean"] = (tcol, "mean")
    spec["Time_Std"] = (tcol, "std")
    spec["Time_Total"] = (tcol, "sum")
    for extra in ("k", "k_max", "window", "k_mean"):
        if extra in folds_df.columns:
            spec[extra] = (extra, "first")
    summary = folds_df.groupby(keys, as_index=False).agg(**spec)
    for metric in _CV_METRICS:
        summary[metric] = [f"{m:.4f} ± {s:.4f}" for m, s in
                           zip(summary[f"{metric}_Mean"], summary[f"{metric}_Std"].fillna(0.0))]
    summary[tcol] = [f"{m:.4f} ± {s:.4f}" for m, s in
                     zip(summary["Time_Mean"], summary["Time_Std"].fillna(0.0))]
    return summary


def safe_append_cv_excel(output_file, rows, lock, keys):
    """Append fold rows and rebuild the summary. One append per finished task."""
    with lock:
        try:
            new_df = pd.DataFrame(rows)
            if os.path.exists(output_file):
                existing = pd.read_excel(output_file, sheet_name="Fold_Scores")
                folds_df = pd.concat([existing, new_df], ignore_index=True)
            else:
                folds_df = new_df
            summary_df = summarise_cv(folds_df, keys)
            temp_file = output_file.replace('.xlsx', '_temp.xlsx')
            with pd.ExcelWriter(temp_file, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                folds_df.to_excel(writer, sheet_name="Fold_Scores", index=False)
            if os.path.exists(output_file):
                os.remove(output_file)
            shutil.move(temp_file, output_file)
            return True
        except Exception as exc:
            print(f"✗ Error writing {os.path.basename(output_file)}: {exc}")
            return False


def completed_cv_tasks(output_file, keys):
    """Task keys already in the workbook. A task writes atomically, so present
    means finished."""
    done = set()
    if not output_file or not os.path.exists(output_file):
        return done
    try:
        df = pd.read_excel(output_file, sheet_name="Fold_Scores")
        for row in df[keys].itertuples(index=False):
            done.add(tuple(str(v) for v in row))
    except Exception as exc:
        print(f"⚠ Could not read {os.path.basename(output_file)}: {exc}")
    return done


def run_cv_phase(title, output_prefix, tasks, worker, keys, workers=None):
    """Shared driver: resume, dispatch, report. `tasks` are worker arguments."""
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")
    existing = get_latest_file(output_prefix)
    if existing:
        output_file = existing
        print(f"📂 Resuming from: {output_file}\n")
    else:
        timestamp = datetime.now().strftime("%y_%m_%d_%H%M%S")
        output_file = results_path(f"{output_prefix}{timestamp}.xlsx")
        print(f"📝 Creating: {output_file}\n")

    done = completed_cv_tasks(output_file, keys)
    pending = [t for t in tasks if tuple(str(v) for v in t[0]) not in done]
    total = len(tasks)
    print(f"Total tasks: {total}\nAlready done: {len(done)}\nRemaining: {len(pending)}\n")
    if not pending:
        print("✓ Already complete!\n")
        return output_file

    manager = _MP_CTX.Manager()
    lock = manager.Lock()
    count = total - len(pending)
    with ProcessPoolExecutor(max_workers=workers or MAX_WORKERS,
                             mp_context=_MP_CTX) as executor:
        futures = {executor.submit(worker, args, output_file, lock, keys): args
                   for _, args in pending}
        for future in as_completed(futures):
            if future.result():
                count += 1
            print(f"Progress: {count}/{total} ({100*count/max(total,1):.1f}%)")
    print(f"\n✓ Complete! Results: {output_file}\n")
    return output_file


# ============================================================================
# PHASE 1: FIXED k, SCORED BY CROSS-VALIDATION
# ============================================================================

def cv_fixed_k_task(args, output_file, lock, keys):
    dataset_path, dataset_name, model_name, k, noise_rate = args
    try:
        X, y = load_pooled_dataset(dataset_path)
        if X is None:
            print(f"⚠ Missing data: {dataset_name}")
            return None
        design = cv_design(X, y, CV_EVAL_FOLDS, CV_EVAL_REPEATS, CV_SEED)
        if design is None:
            print(f"✗ {dataset_name}: cannot stratify")
            return None
        X, y, splits, n_splits, note = design
        model_class = MODEL_MAP[model_name]

        rows = []
        for repeat, fold, tr, te in splits:
            X_tr, X_te = scale_fold(X[tr], X[te])
            y_tr, flipped = fold_labels(y[tr], dataset_name, repeat, fold, noise_rate)
            if k >= len(X_tr):
                continue
            t = Timer(); t.start()
            model = model_class(k=int(k))
            model.fit(X_tr, y_tr)
            predictions = model.predict(X_te)
            elapsed = t.stop()
            rows.append({"Dataset": dataset_name, "Model": model_name, "k": int(k),
                         "Repeat": repeat, "Fold": fold, "Folds_Used": n_splits,
                         "N_Train": len(tr), "N_Test": len(te),
                         "Labels_Flipped": flipped, "Stratification_Note": note,
                         **fold_metrics(y[te], predictions, elapsed)})
        if not rows:
            return None
        accs = [r["Accuracy"] for r in rows]
        safe_append_cv_excel(output_file, rows, lock, keys)
        print(f"✓ Phase 1: {dataset_name:12s} | {model_name:13s} | k={k:<3d} | "
              f"Acc={np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}")
        return rows
    except Exception as exc:
        print(f"✗ Error: {dataset_name} | {model_name} | k={k} -> {exc}")
        return None


def phase1_initial_experiments(base_folder, k_values):
    keys = ["Dataset", "Model", "k"]
    tasks = []
    for dataset_name in list_datasets(base_folder):
        path = os.path.join(base_folder, dataset_name)
        for k in k_values:
            for model_name in MODELS:
                tasks.append(((dataset_name, model_name, k),
                              (path, dataset_name, model_name, k, ACTIVE_NOISE_RATE)))
    return run_cv_phase(
        f"PHASE 1: FIXED k={', '.join(str(k) for k in k_values)} "
        f"({CV_EVAL_REPEATS}x{CV_EVAL_FOLDS}-fold CV)",
        "results_phase1_initial_", tasks, cv_fixed_k_task, keys)


# ============================================================================
# PHASE 2: BEST k BY STRATIFIED GRID SEARCH
#
#   The equivalent of GridSearchCV over k, scored by repeated stratified CV.
#   Its best score is the SELECTION criterion, not a performance estimate: the
#   k that wins was chosen by looking at these same folds, so reporting its
#   score as accuracy is optimistic.  Phase 4 exists to give the unbiased
#   number, by re-running the whole search inside every outer fold.
# ============================================================================

def cv_grid_search_task(args, output_file, lock, keys):
    dataset_path, dataset_name, model_name, k_range, noise_rate = args
    try:
        X, y = load_pooled_dataset(dataset_path)
        if X is None:
            return None
        design = cv_design(X, y, CV_BESTK_FOLDS, CV_BESTK_REPEATS, CV_SEED)
        if design is None:
            print(f"✗ {dataset_name}: cannot stratify")
            return None
        X, y, splits, n_splits, note = design
        model_class = MODEL_MAP[model_name]

        rows = []
        for repeat, fold, tr, te in splits:
            X_tr, X_te = scale_fold(X[tr], X[te])
            y_tr, flipped = fold_labels(y[tr], dataset_name, repeat, fold, noise_rate)
            usable = [int(k) for k in k_range if k < len(X_tr)]
            t = Timer(); t.start()
            for k in usable:
                model = model_class(k=k)
                model.fit(X_tr, y_tr)
                rows.append({"Dataset": dataset_name, "Model": model_name, "k": k,
                             "Repeat": repeat, "Fold": fold, "Folds_Used": n_splits,
                             "N_Train": len(tr), "N_Test": len(te),
                             "Labels_Flipped": flipped, "Stratification_Note": note,
                             **fold_metrics(y[te], model.predict(X_te), 0.0)})
            elapsed = t.stop()
            for row in rows[-len(usable):]:
                row[Timer.label()] = elapsed / max(len(usable), 1)
        if not rows:
            return None
        safe_append_cv_excel(output_file, rows, lock, keys)
        table = pd.DataFrame(rows).groupby("k")["Accuracy"].mean()
        best = table.max()
        tied = [int(k) for k, v in table.items() if v == best]
        best_k = max(tied) if CV_TIE_BREAKING == "highest_k" else min(tied)
        print(f"✓ Phase 2: {dataset_name:12s} | {model_name:13s} → best k={best_k} "
              f"(CV {best:.4f}, {len(tied)} tied)")
        return rows
    except Exception as exc:
        print(f"✗ Error: {dataset_name} | {model_name} -> {exc}")
        return None


def phase2_find_best_k(base_folder, k_range):
    keys = ["Dataset", "Model", "k"]
    tasks = []
    for dataset_name in list_datasets(base_folder):
        path = os.path.join(base_folder, dataset_name)
        for model_name in MODELS:
            tasks.append(((dataset_name, model_name, min(k_range)),
                          (path, dataset_name, model_name, list(k_range), ACTIVE_NOISE_RATE)))
    output_file = run_cv_phase(
        f"PHASE 2: BEST k BY GRID SEARCH ({CV_BESTK_REPEATS}x{CV_BESTK_FOLDS}-fold "
        f"stratified CV, k={min(k_range)}-{max(k_range)})",
        "results_phase2_best_k_", tasks, cv_grid_search_task, keys)
    return best_k_from_grid(output_file)


def best_k_from_grid(output_file):
    """
    {(dataset, model): best_k} plus a Best_k_Summary sheet, from the grid the
    phase scored.  Ties go to the smallest k unless configured otherwise —
    on flat curves dozens of k score identically, and the tie count is written
    out so the choice cannot be mistaken for a sharp optimum.
    """
    if not output_file or not os.path.exists(output_file):
        return {}
    folds = pd.read_excel(output_file, sheet_name="Fold_Scores")
    grid = folds.groupby(["Dataset", "Model", "k"])["Accuracy"].agg(["mean", "std", "size"])
    grid = grid.rename(columns={"mean": "CV_Accuracy", "std": "CV_Std", "size": "Folds"})
    # What the whole search cost, so the timing report can charge tuning to Phase 2.
    tcol = _fold_time_column(folds) or Timer.label()
    search_cost = folds.groupby(["Dataset", "Model"])[tcol].sum()
    best_k, summary_rows = {}, []
    for (dataset, model), sub in grid.groupby(level=[0, 1]):
        scores = sub.droplevel([0, 1])["CV_Accuracy"]
        top = scores.max()
        tied = sorted(int(k) for k, v in scores.items() if v == top)
        chosen = max(tied) if CV_TIE_BREAKING == "highest_k" else min(tied)
        best_k[(dataset, model)] = chosen
        within = sorted(int(k) for k, v in scores.items() if v >= top - 0.01)
        summary_rows.append({
            "Dataset": dataset, "Model": model, "Best_k": chosen,
            "CV_Accuracy": top, "CV_Std": float(sub.droplevel([0, 1]).loc[chosen, "CV_Std"]),
            "Folds": int(sub.droplevel([0, 1]).loc[chosen, "Folds"]),
            "Tied_k_count": len(tied), "Tied_k": ", ".join(map(str, tied[:15])),
            "k_within_1pct_count": len(within),
            Timer.label(): float(search_cost.get((dataset, model), 0.0)),
        })
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a',
                        if_sheet_exists='replace') as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Best_k_Summary", index=False)
        grid.reset_index().to_excel(writer, sheet_name="All_CV_Scores", index=False)
    return best_k


# ============================================================================
# PHASE 3: THE TUNED k, SCORED BY CROSS-VALIDATION
#
#   Reported for continuity with the fixed-k and adaptive phases.  Because the
#   k comes from Phase 2's search over the same data, this number is optimistic
#   by construction — Phase 4 is the one to quote as performance.
# ============================================================================

def cv_best_k_task(args, output_file, lock, keys):
    dataset_path, dataset_name, model_name, best_k, noise_rate = args
    try:
        X, y = load_pooled_dataset(dataset_path)
        if X is None:
            return None
        design = cv_design(X, y, CV_EVAL_FOLDS, CV_EVAL_REPEATS, CV_SEED)
        if design is None:
            return None
        X, y, splits, n_splits, note = design
        model_class = MODEL_MAP[model_name]

        rows = []
        for repeat, fold, tr, te in splits:
            X_tr, X_te = scale_fold(X[tr], X[te])
            y_tr, flipped = fold_labels(y[tr], dataset_name, repeat, fold, noise_rate)
            k = min(int(best_k), len(X_tr) - 1)
            t = Timer(); t.start()
            model = model_class(k=k)
            model.fit(X_tr, y_tr)
            predictions = model.predict(X_te)
            elapsed = t.stop()
            rows.append({"Dataset": dataset_name, "Model": model_name, "k": k,
                         "Repeat": repeat, "Fold": fold, "Folds_Used": n_splits,
                         "N_Train": len(tr), "N_Test": len(te),
                         "Labels_Flipped": flipped, "Stratification_Note": note,
                         **fold_metrics(y[te], predictions, elapsed)})
        if not rows:
            return None
        accs = [r["Accuracy"] for r in rows]
        safe_append_cv_excel(output_file, rows, lock, keys)
        print(f"✓ Phase 3: {dataset_name:12s} | {model_name:13s} | best k={best_k} | "
              f"Acc={np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}")
        return rows
    except Exception as exc:
        print(f"✗ Error: {dataset_name} | {model_name} -> {exc}")
        return None


def phase3_test_best_k(base_folder, best_k_dict):
    keys = ["Dataset", "Model"]
    tasks = []
    for dataset_name in list_datasets(base_folder):
        path = os.path.join(base_folder, dataset_name)
        for model_name in MODELS:
            best_k = best_k_dict.get((dataset_name, model_name))
            if best_k is None:
                continue
            tasks.append(((dataset_name, model_name),
                          (path, dataset_name, model_name, best_k, ACTIVE_NOISE_RATE)))
    if not tasks:
        print("\n⚠ Phase 3 skipped — no best k available. Run Phase 2 first.\n")
        return None
    return run_cv_phase(
        f"PHASE 3: TUNED k ({CV_EVAL_REPEATS}x{CV_EVAL_FOLDS}-fold CV)",
        "results_phase3_best_k_test_", tasks, cv_best_k_task, keys)


# ============================================================================
# ADAPTIVE HELPERS
# ============================================================================

def compute_distance_matrix(X):
    """Vectorized pairwise Euclidean distance matrix. Uses GPU if backend is 'gpu'."""
    if get_backend() == "gpu":
        from knn_algorithms import _batched_cdist, _get_device
        device = _get_device()
        return _batched_cdist(np.asarray(X, dtype=np.float32),
                              np.asarray(X, dtype=np.float32), device)
    from scipy.spatial.distance import cdist
    return cdist(X, X, metric='euclidean').astype(np.float32)


def compute_first_correct_k(X, y, model_name, max_k):
    """
    For each training point, find the smallest k (1..max_k) that classifies
    it correctly via leave-one-out.

    Fully vectorized: computes predictions for ALL points × ALL k values
    using matrix operations instead of Python loops.
    """
    n = len(y)
    first_correct_k = np.full(n, max_k, dtype=int)

    dist_matrix = compute_distance_matrix(X)

    # sorted_neighbors[i] = indices of all other points sorted by distance from i
    sorted_neighbors = np.argsort(dist_matrix, axis=1)[:, 1:]  # exclude self

    # Precompute kth neighbor distances: kth_dist[i, k-1] = dist from i to k-th NN
    kth_dist = np.empty((n, max_k), dtype=np.float32)
    for k_idx in range(max_k):
        kth_dist[:, k_idx] = dist_matrix[np.arange(n), sorted_neighbors[:, k_idx]]

    # Get unique classes
    classes = np.unique(y)
    n_classes = len(classes)

    for k in range(1, max_k + 1):
        # Only compute for points that haven't found their correct k yet
        mask_undecided = first_correct_k == max_k
        if not np.any(mask_undecided):
            break  # all points found their k, skip remaining

        if model_name == "CustomKNN":
            # Vectorized: gather labels of first k neighbors for ALL points at once
            voter_labels = y[sorted_neighbors[:, :k]]  # (n, k)
            # Count votes per class for each point
            preds = _vectorized_majority(voter_labels, classes, n)

        elif model_name == "MutualKNN":
            voter_labels = y[sorted_neighbors[:, :k]]  # (n, k)
            # For each point i and each of its k neighbors j:
            # keep j only if dist(i,j) <= kth_dist[j, k-1]
            k_neighbor_indices = sorted_neighbors[:, :k]  # (n, k)
            # dist(i, j) for each i and each of its k neighbors
            i_indices = np.arange(n)[:, None]  # (n, 1)
            dists_to_neighbors = dist_matrix[i_indices, k_neighbor_indices]  # (n, k)
            # kth dist of each neighbor j
            kth_of_neighbors = kth_dist[k_neighbor_indices, k - 1]  # (n, k)
            # mutual mask: True where neighbor is mutual
            mutual_mask = dists_to_neighbors <= kth_of_neighbors  # (n, k)
            # For points with no mutual neighbors, fall back to all k neighbors
            has_mutual = np.any(mutual_mask, axis=1)  # (n,)
            # Build effective mask: mutual where available, all k otherwise
            effective_mask = np.where(has_mutual[:, None], mutual_mask, True)
            # Masked majority vote
            preds = _vectorized_masked_majority(voter_labels, effective_mask, classes, n)

        elif model_name == "SymmetricKNN":
            # Avoid n×n boolean masks — compute vote counts in row batches.
            threshold = kth_dist[:, k - 1]  # (n,) k-th NN dist per training point

            # Forward vote counts: for each point i, count labels of its k nearest neighbors
            forward_labels = y[sorted_neighbors[:, :k]]  # (n, k)
            forward_counts = np.zeros((n, n_classes), dtype=int)
            for ci, c in enumerate(classes):
                forward_counts[:, ci] = np.sum(forward_labels == c, axis=1)

            # Reverse vote counts: for each point i, count labels of points j
            # where dist(i,j) <= threshold[j] (i.e., i is within j's k-NN ball).
            # Process in batches to avoid n×n memory allocation.
            BATCH = max(1, min(4096, 500_000_000 // max(n, 1)))  # ~500MB limit
            reverse_counts = np.zeros((n, n_classes), dtype=int)
            for start in range(0, n, BATCH):
                end = min(start + BATCH, n)
                # (batch, n) boolean: which training points j are reverse neighbors of batch points
                rev_mask = dist_matrix[start:end] <= threshold[None, :]  # broadcast (1,n)
                # zero out self
                batch_idx = np.arange(start, end)
                rev_mask[np.arange(end - start), batch_idx] = False
                # Count per class: (batch, n) & (1, n) for each class, sum over j
                for ci, c in enumerate(classes):
                    reverse_counts[start:end, ci] = np.sum(
                        rev_mask & (y[None, :] == c), axis=1)

            # Union: a voter is counted if it appears in forward OR reverse.
            # Since a voter might appear in both, we can't just sum.
            # But for majority vote, we need union counts.
            # Trick: total_unique_votes[i,c] = forward[i,c] + reverse[i,c] - overlap[i,c]
            # Overlap: points that are both forward AND reverse neighbors.
            # For speed, use the max approach: union >= max(forward, reverse) per class.
            # Actually: forward counts k-neighbors' labels, reverse counts reverse-neighbors' labels.
            # A point j could be in both sets, so we'd double count.
            #
            # Correct approach: compute union counts directly in batches.
            # union_counts[i, c] = number of unique voters for i with label c
            # = |{j : (j in forward_k(i) OR dist(i,j)<=threshold[j]) AND y[j]==c}|

            union_counts = np.zeros((n, n_classes), dtype=int)
            for start in range(0, n, BATCH):
                end = min(start + BATCH, n)
                batch_size = end - start
                batch_idx = np.arange(start, end)

                # Forward mask for batch: (batch, n)
                fwd = np.zeros((batch_size, n), dtype=bool)
                np.put_along_axis(fwd, sorted_neighbors[start:end, :k], True, axis=1)

                # Reverse mask for batch: (batch, n)
                rev = dist_matrix[start:end] <= threshold[None, :]
                rev[np.arange(batch_size), batch_idx] = False

                # Union
                union = fwd | rev

                for ci, c in enumerate(classes):
                    union_counts[start:end, ci] = np.sum(
                        union & (y[None, :] == c), axis=1)

                del fwd, rev, union

            preds = classes[np.argmax(union_counts, axis=1)]

        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Update first_correct_k for points that are correct at this k and still undecided
        correct_at_k = (preds == y) & mask_undecided
        first_correct_k[correct_at_k] = k

    return first_correct_k


def _vectorized_majority(voter_labels, classes, n):
    """
    Given voter_labels (n, k) — labels of k voters for each of n points,
    return the majority vote for each point.
    """
    vote_counts = np.zeros((n, len(classes)), dtype=int)
    for ci, c in enumerate(classes):
        vote_counts[:, ci] = np.sum(voter_labels == c, axis=1)
    return classes[np.argmax(vote_counts, axis=1)]


def _vectorized_masked_majority(voter_labels, mask, classes, n):
    """
    Given voter_labels (n, k) and mask (n, k) — True for voters that count,
    return the majority vote for each point (only counting masked-in voters).
    """
    vote_counts = np.zeros((n, len(classes)), dtype=int)
    for ci, c in enumerate(classes):
        vote_counts[:, ci] = np.sum((voter_labels == c) & mask, axis=1)
    return classes[np.argmax(vote_counts, axis=1)]

def compute_adaptive_k_last_stable(X, y, model_name, max_k, stability_window=3,
                                   return_correctness=False):
    """
    For each training point, find the last stable k: the end of the first
    window of `stability_window` consecutive k values that all predict correctly.
    Fully vectorized — no Python loops over data points.
    """
    n = len(y)
    adaptive_k = np.full(n, max_k, dtype=int)

    dist_matrix = compute_distance_matrix(X)
    sorted_neighbors = np.argsort(dist_matrix, axis=1)[:, 1:]

    kth_dist = np.empty((n, max_k), dtype=np.float32)
    for k_idx in range(max_k):
        kth_dist[:, k_idx] = dist_matrix[np.arange(n), sorted_neighbors[:, k_idx]]

    classes = np.unique(y)
    n_classes = len(classes)

    # Build correctness matrix: correct[i, k-1] = True if prediction at k is correct for point i
    correct = np.zeros((n, max_k), dtype=bool)

    for k in range(1, max_k + 1):
        if model_name == "CustomKNN":
            voter_labels = y[sorted_neighbors[:, :k]]
            preds = _vectorized_majority(voter_labels, classes, n)

        elif model_name == "MutualKNN":
            voter_labels = y[sorted_neighbors[:, :k]]
            k_neighbor_indices = sorted_neighbors[:, :k]
            i_indices = np.arange(n)[:, None]
            dists_to_neighbors = dist_matrix[i_indices, k_neighbor_indices]
            kth_of_neighbors = kth_dist[k_neighbor_indices, k - 1]
            mutual_mask = dists_to_neighbors <= kth_of_neighbors
            has_mutual = np.any(mutual_mask, axis=1)
            effective_mask = np.where(has_mutual[:, None], mutual_mask, True)
            preds = _vectorized_masked_majority(voter_labels, effective_mask, classes, n)

        elif model_name == "SymmetricKNN":
            threshold = kth_dist[:, k - 1]
            BATCH = max(1, min(4096, 500_000_000 // max(n, 1)))
            union_counts = np.zeros((n, n_classes), dtype=int)
            for start in range(0, n, BATCH):
                end = min(start + BATCH, n)
                batch_size = end - start
                batch_idx = np.arange(start, end)

                fwd = np.zeros((batch_size, n), dtype=bool)
                np.put_along_axis(fwd, sorted_neighbors[start:end, :k], True, axis=1)

                rev = dist_matrix[start:end] <= threshold[None, :]
                rev[np.arange(batch_size), batch_idx] = False

                union = fwd | rev
                for ci, c in enumerate(classes):
                    union_counts[start:end, ci] = np.sum(
                        union & (y[None, :] == c), axis=1)
                del fwd, rev, union

            preds = classes[np.argmax(union_counts, axis=1)]

        else:
            raise ValueError(f"Unknown model: {model_name}")

        correct[:, k - 1] = (preds == y)

    if return_correctness:
        return correct
    return scan_stable_window(correct, stability_window)


def scan_stable_window(correct, stability_window):
    """
    End of the first run of `stability_window` consecutive correct predictions.

    Split out from the matrix above because that matrix does not depend on the
    window at all — only this scan does.  Sweeping w therefore costs one
    O(n^2 * k_max) pass in total rather than one per window.
    """
    n, max_k = correct.shape
    adaptive_k = np.full(n, max_k, dtype=int)
    for i in range(n):
        for k_start in range(max_k - stability_window + 1):
            if np.all(correct[i, k_start:k_start + stability_window]):
                adaptive_k[i] = k_start + stability_window   # 1-indexed end of window
                break
    return adaptive_k


# ============================================================================
# ADAPTIVE NEIGHBOURHOODS UNDER CROSS-VALIDATION
#
#   AdaNN's per-row k is a FITTED quantity: it comes from leave-one-out over the
#   training set, exactly like a model's weights.  Under cross-validation it has
#   to be derived inside every fold, from that fold's training rows alone.
#
#   Deriving it once over the whole dataset — which is what the old "create
#   dataset" phases did, writing a data1 with an extra column — lets every test
#   row help decide the k of the neighbours that will later classify it.  The
#   leak sits one level below a badly chosen global k, which is what makes it
#   easy to miss in review.
#
#   Phases 5/7/9 derive and cache the per-fold k vectors; Phases 6/8/10 score
#   them.  The halves stay apart so the cost of BUILDING a neighbourhood remains
#   separable from the cost of USING it — the breakdown the computational-cost
#   questions are really asking for.
# ============================================================================

# kind -> (label, results prefix for the derive phase, prefix for the score phase)
ADAPTIVE_KINDS = {
    "adaptive": ("fixed k_max", "results_phase5_adaptive_timing_",
                 "results_phase6_adaptive_test_"),
    "sqrt":     ("k_max = sqrt(n/2)", "results_phase7_sqrt_timing_",
                 "results_phase8_sqrt_test_"),
    "lastk":    ("last stable k", "results_phase9_last_k_timing_",
                 "results_phase10_last_k_test_"),
}


def adaptive_budget(kind, param, n_train):
    """The k_max this fold may search up to, capped by the training rows."""
    if kind == "adaptive":
        budget = int(param)
    elif kind == "sqrt":
        budget = math.ceil(math.sqrt(n_train / 2))
    else:
        budget = LAST_K_MAX
    return max(1, min(int(budget), n_train - 1))


def adaptive_k_vector(kind, X_train, y_train, model_name, k_max, param=None):
    """For "lastk", `param` is the stability window; the other kinds ignore it."""
    if kind == "lastk":
        window = STABILITY_WINDOW if param is None else int(param)
        return compute_adaptive_k_last_stable(X_train, y_train, model_name,
                                              k_max, window)
    return compute_first_correct_k(X_train, y_train, model_name, k_max)


def adaptive_cache_root(kind, param):
    if kind == "adaptive":
        return adaptive_folder(param)
    if kind == "sqrt":
        return sqrt_folder()
    return last_k_folder(param)


def adaptive_cache_path(kind, param, model_name, dataset_name):
    return os.path.join(adaptive_cache_root(kind, param), model_name,
                        dataset_name, "folds.npz")


def derive_adaptive_task(args):
    """Derive one (dataset, model, config) pair's per-fold k vectors and cache them."""
    dataset_path, dataset_name, model_name, kind, param, noise_rate = args
    cache = adaptive_cache_path(kind, param, model_name, dataset_name)
    if os.path.exists(cache):
        print(f"⊘ Skipping (cached): {dataset_name} | {model_name} | {kind}")
        return None
    try:
        X, y = load_pooled_dataset(dataset_path)
        if X is None:
            print(f"⚠ Missing data: {dataset_name}")
            return None
        design = cv_design(X, y, CV_EVAL_FOLDS, CV_EVAL_REPEATS, CV_SEED)
        if design is None:
            print(f"✗ {dataset_name}: cannot stratify")
            return None
        X, y, splits, n_splits, _ = design

        store, budgets, total = {}, {}, 0.0
        for repeat, fold, tr, te in splits:
            X_tr, _ = scale_fold(X[tr], X[te])
            y_tr, _ = fold_labels(y[tr], dataset_name, repeat, fold, noise_rate)
            budget = adaptive_budget(kind, param, len(X_tr))
            t = Timer(); t.start()
            store[f"r{repeat}f{fold}"] = adaptive_k_vector(kind, X_tr, y_tr,
                                                           model_name, budget, param)
            total += t.stop()
            budgets[f"b{repeat}_{fold}"] = budget

        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, **store, **budgets)
        mean_k = float(np.mean([v.mean() for v in store.values()]))
        print(f"✓ {dataset_name:12s} | {model_name:13s} | {kind:8s} | "
              f"{len(splits)} folds | k̄={mean_k:.1f} | {total:.1f}s")
        record = {"Dataset": dataset_name, "Model": model_name,
                  "k_max": adaptive_budget(kind, param, len(X) - len(X) // n_splits),
                  "Folds": len(splits), Timer.label(): total}
        if kind == "lastk":
            # The timing report joins Phase 9 to Phase 10 on the window as well.
            record["window"] = int(param)
        return record
    except Exception as exc:
        print(f"✗ Error: {dataset_name} | {model_name} | {kind} -> {exc}")
        return None


def run_derive_phase(kind, param, title):
    """Phases 5/7/9 — derive the per-fold neighbourhoods, sequentially (RAM heavy)."""
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")
    _, timing_prefix, _ = ADAPTIVE_KINDS[kind]
    tasks = [(os.path.join(BASE_FOLDER, d), d, m, kind, param, ACTIVE_NOISE_RATE)
             for m in MODELS for d in list_datasets(BASE_FOLDER)]
    pending = [t for t in tasks
               if not os.path.exists(adaptive_cache_path(kind, param, t[2], t[1]))]
    print(f"Total tasks: {len(tasks)}\nRemaining: {len(pending)}\n")
    if not pending:
        print("✓ Already complete!\n")
        return None
    records = []
    for i, task in enumerate(pending, 1):
        record = derive_adaptive_task(task)
        if record:
            records.append(record)
        print(f"Progress: {i}/{len(pending)} ({100*i/len(pending):.1f}%)")
    if not records:
        return None
    timing_file = merge_timing_csv(timing_prefix, records, ["Dataset", "Model", "k_max"])
    print(f"\n✓ Complete! Timing: {timing_file}\n")
    return timing_file


def score_adaptive_task(args, output_file, lock, keys):
    """Phases 6/8/10 — score the cached neighbourhoods over the same folds.

    The cache path arrives as an argument rather than being rebuilt here. It
    depends on VARIANT_DIR, which activate_variant() sets in the PARENT only;
    with backend "gpu" the pool uses "spawn", so a worker re-imports this module
    with VARIANT_DIR back at its default "" and looked for the noisy caches in
    the clean location. Every task then reported "not derived yet" even though
    Phase 5 had just written them.
    """
    dataset_path, dataset_name, model_name, kind, param, noise_rate, cache = args
    try:
        if not os.path.exists(cache):
            print(f"⚠ Not derived yet: {dataset_name} | {model_name} | {kind}")
            return None
        X, y = load_pooled_dataset(dataset_path)
        design = cv_design(X, y, CV_EVAL_FOLDS, CV_EVAL_REPEATS, CV_SEED)
        if design is None:
            return None
        X, y, splits, n_splits, note = design
        model_class = MODEL_MAP[model_name]
        cached = np.load(cache)

        rows = []
        for repeat, fold, tr, te in splits:
            key = f"r{repeat}f{fold}"
            if key not in cached:
                continue
            k_per_row = cached[key]
            X_tr, X_te = scale_fold(X[tr], X[te])
            y_tr, flipped = fold_labels(y[tr], dataset_name, repeat, fold, noise_rate)
            t = Timer(); t.start()
            predictions, ks_used = predict_with_adaptive_k(model_class, X_tr, y_tr,
                                                           X_te, k_per_row)
            elapsed = t.stop()
            rows.append({
                "Dataset": dataset_name, "Model": model_name, "Kind": kind,
                "k_max": int(adaptive_budget(kind, param, len(tr))),
                "window": int(param) if kind == "lastk" else np.nan,
                "Repeat": repeat, "Fold": fold, "Folds_Used": n_splits,
                "N_Train": len(tr), "N_Test": len(te),
                "Labels_Flipped": flipped, "Stratification_Note": note,
                "k_mean": float(np.mean(ks_used)), "k_std": float(np.std(ks_used)),
                "k_min": int(np.min(ks_used)), "k_max_used": int(np.max(ks_used)),
                **fold_metrics(y[te], predictions, elapsed)})
        if not rows:
            return None
        accs = [r["Accuracy"] for r in rows]
        safe_append_cv_excel(output_file, rows, lock, keys)
        print(f"✓ {dataset_name:12s} | {model_name:13s} | {kind:8s} | "
              f"Acc={np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}")
        return rows
    except Exception as exc:
        print(f"✗ Error: {dataset_name} | {model_name} | {kind} -> {exc}")
        return None


def run_score_phase(kind, params, title):
    """Phases 6/8/10 — one task per (dataset, model, config)."""
    _, _, prefix = ADAPTIVE_KINDS[kind]
    # The window separates one lastk sweep value from another, so it has to be
    # part of the resume key or the second window would look already finished.
    keys = ["Dataset", "Model", "k_max", "window"] if kind == "lastk" \
        else ["Dataset", "Model", "k_max"]
    tasks = []
    for param in params:
        for dataset_name in list_datasets(BASE_FOLDER):
            path = os.path.join(BASE_FOLDER, dataset_name)
            for model_name in MODELS:
                if kind == "adaptive":
                    key = (dataset_name, model_name, param)
                elif kind == "lastk":
                    key = (dataset_name, model_name, LAST_K_MAX, int(param))
                else:
                    key = (dataset_name, model_name, "sqrt")
                tasks.append((key, (path, dataset_name, model_name, kind, param,
                                    ACTIVE_NOISE_RATE,
                                    adaptive_cache_path(kind, param, model_name,
                                                        dataset_name))))
    return run_cv_phase(title, prefix, tasks, score_adaptive_task, keys,
                        workers=TESTER_WORKERS)


# ============================================================================
# PHASE 5 / 6: FIXED k_max
# ============================================================================

def phase5_create_adaptive_datasets():
    files = [run_derive_phase("adaptive", k_max,
                              f"PHASE 5: DERIVE ADAPTIVE NEIGHBOURHOODS (k_max={k_max}, "
                              f"{CV_EVAL_REPEATS}x{CV_EVAL_FOLDS}-fold)")
             for k_max in ADAPTIVE_K_MAX_VALUES]
    return next((f for f in reversed(files) if f), None)


def phase6_test_adaptive_datasets():
    return run_score_phase("adaptive", ADAPTIVE_K_MAX_VALUES,
                           f"PHASE 6: SCORE ADAPTIVE (k_max={ADAPTIVE_K_MAX_VALUES}, "
                           f"{CV_EVAL_REPEATS}x{CV_EVAL_FOLDS}-fold CV)")


# ============================================================================
# PHASE 7 / 8: k_max = sqrt(n/2)
# ============================================================================

def phase7_create_sqrt_datasets():
    return run_derive_phase("sqrt", None,
                            f"PHASE 7: DERIVE SQRT(N/2) NEIGHBOURHOODS "
                            f"({CV_EVAL_REPEATS}x{CV_EVAL_FOLDS}-fold)")


def phase8_test_sqrt_datasets():
    return run_score_phase("sqrt", [None],
                           f"PHASE 8: SCORE SQRT(N/2) "
                           f"({CV_EVAL_REPEATS}x{CV_EVAL_FOLDS}-fold CV)")


# ============================================================================
# PHASE 9 / 10: LAST STABLE k
# ============================================================================

def phase9_create_last_k_datasets():
    files = [run_derive_phase("lastk", window,
                              f"PHASE 9: DERIVE LAST-STABLE-k (k_max={LAST_K_MAX}, "
                              f"window={window}, "
                              f"{CV_EVAL_REPEATS}x{CV_EVAL_FOLDS}-fold)")
             for window in STABILITY_WINDOWS]
    return next((f for f in reversed(files) if f), None)


def phase10_test_last_k_datasets():
    return run_score_phase("lastk", STABILITY_WINDOWS,
                           f"PHASE 10: SCORE LAST-STABLE-k (windows={STABILITY_WINDOWS}, "
                           f"{CV_EVAL_REPEATS}x{CV_EVAL_FOLDS}-fold CV)")


# ============================================================================
# PHASE 11: FRIEDMAN TEST
# ============================================================================

def collect_all_results(value="Accuracy_Mean"):
    """
    One number per (method, dataset), gathered from the phases that score models.

    Every phase now writes a Summary sheet of fold means, so the matrix the
    Friedman test ranks is built from fold-averaged accuracy rather than from a
    single split.  `value` selects which column to pull, so the same walk serves
    the ranking (Accuracy_Mean) and the timing report (Time_Mean).
    """
    sources = [
        ("results_phase1_initial_", "Phase 1",
         lambda r: f"{r['Model']}_k{int(r['k'])}"),
        ("results_phase3_best_k_test_", "Phase 3",
         lambda r: f"{r['Model']}_BestK"),
        ("results_phase6_adaptive_test_", "Phase 6",
         lambda r: f"{r['Model']}_Adaptive_k{int(r['k_max'])}"),
        ("results_phase8_sqrt_test_", "Phase 8",
         lambda r: f"{r['Model']}_Sqrt_n2"),
        ("results_phase10_last_k_test_", "Phase 10",
         lambda r: f"{r['Model']}_LastK_w{int(r['window'])}"),
    ]
    results = {}
    for prefix, label, namer in sources:
        path = get_latest_file(prefix)
        if not path or not os.path.exists(path):
            continue
        try:
            frame = pd.read_excel(path, sheet_name="Summary")
        except Exception as exc:
            print(f"⚠ {label}: {exc}")
            continue
        if value not in frame.columns:
            print(f"⚠ {label}: no '{value}' column — skipping")
            continue
        print(f"📂 Reading {label}: {path}")
        for _, row in frame.iterrows():
            try:
                method = namer(row)
            except Exception:
                continue
            results.setdefault(method, {})[row["Dataset"]] = row[value]
    return results


def create_accuracy_matrix(results):
    """Create matrix: rows=datasets, columns=methods."""
    all_datasets = set()
    all_methods = set()
    for method, datasets in results.items():
        all_methods.add(method)
        all_datasets.update(datasets.keys())
    all_datasets = sorted(list(all_datasets))
    all_methods = sorted(list(all_methods))
    print(f"\n📊 Found {len(all_datasets)} datasets and {len(all_methods)} methods")
    matrix = []
    for dataset in all_datasets:
        row = []
        for method in all_methods:
            if dataset in results.get(method, {}):
                row.append(results[method][dataset])
            else:
                row.append(np.nan)
        matrix.append(row)
    df = pd.DataFrame(matrix, columns=all_methods, index=all_datasets)
    missing_threshold = 0.5
    complete_methods = []
    for method in all_methods:
        missing_ratio = df[method].isna().sum() / len(all_datasets)
        if missing_ratio <= missing_threshold:
            complete_methods.append(method)
        else:
            print(f"⚠ Removing {method} (missing {missing_ratio*100:.1f}% of datasets)")
    df = df[complete_methods]
    df = df.dropna(axis=0)
    print(f"\n✓ Final matrix: {df.shape[0]} datasets × {df.shape[1]} methods")
    return df

def perform_friedman_test(df):
    """Perform Friedman test."""
    print("\n" + "="*80)
    print("FRIEDMAN TEST")
    print("="*80)
    data = df.values
    method_names = df.columns
    dataset_names = df.index
    print(f"\nDatasets: {len(dataset_names)}")
    print(f"Methods: {len(method_names)}")
    statistic, p_value = stats.friedmanchisquare(*[data[:, i] for i in range(data.shape[1])])
    print("\n" + "="*80)
    print("FRIEDMAN TEST RESULTS")
    print("="*80)
    print(f"Friedman statistic (χ²): {statistic:.4f}")
    print(f"P-value: {p_value:.6e}")
    print(f"Degrees of freedom: {data.shape[1] - 1}")
    if p_value < 0.05:
        print("\n✓ Result: SIGNIFICANT difference between methods (p < 0.05)")
        print("  The methods perform differently across datasets.")
    else:
        print("\n✗ Result: NO significant difference between methods (p >= 0.05)")
        print("  The methods perform similarly across datasets.")
    ranks = np.zeros_like(data, dtype=float)
    for i in range(data.shape[0]):
        ranks[i, :] = stats.rankdata(-data[i, :])
    average_ranks = ranks.mean(axis=0)
    summary = pd.DataFrame({
        'Method': method_names,
        'Average_Rank': average_ranks,
        'Mean_Accuracy': data.mean(axis=0),
        'Std_Accuracy': data.std(axis=0),
        'Min_Accuracy': data.min(axis=0),
        'Max_Accuracy': data.max(axis=0)
    }).sort_values('Average_Rank')
    print("\n" + "="*80)
    print("METHOD RANKINGS (Lower rank = Better performance)")
    print("="*80)
    print(summary.to_string(index=False))
    return summary, statistic, p_value

def save_friedman_results(df, summary, statistic, p_value):
    """Save Friedman results to Excel."""
    timestamp = datetime.now().strftime("%y_%m_%d_%H%M%S")
    output_file = results_path(f"results_phase11_friedman_{timestamp}.xlsx")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Accuracy_Matrix")
        summary.to_excel(writer, sheet_name="Rankings", index=False)
        test_stats = pd.DataFrame({
            'Statistic': ['Friedman χ²', 'P-value', 'Degrees of Freedom', 'Significant (α=0.05)'],
            'Value': [statistic, p_value, df.shape[1] - 1, 'Yes' if p_value < 0.05 else 'No']
        })
        test_stats.to_excel(writer, sheet_name="Test_Statistics", index=False)
        dataset_info = pd.DataFrame({
            'Dataset': df.index,
            'Num_Methods': [df.loc[d].notna().sum() for d in df.index]
        })
        dataset_info.to_excel(writer, sheet_name="Datasets", index=False)
    print(f"\n✓ Results saved to: {output_file}")
    return output_file

def perform_posthoc_nemenyi(df, summary):
    """Perform post-hoc Nemenyi test."""
    print("\n" + "="*80)
    print("POST-HOC NEMENYI TEST")
    print("="*80)
    try:
        from scikit_posthocs import posthoc_nemenyi_friedman
        data = df.values
        nemenyi_results = posthoc_nemenyi_friedman(data)
        nemenyi_results.index = df.columns
        nemenyi_results.columns = df.columns
        print("\nNemenyi p-values (methods with p < 0.05 are significantly different):")
        print(nemenyi_results.round(4))
        print("\n" + "="*80)
        print("SIGNIFICANTLY DIFFERENT PAIRS (p < 0.05)")
        print("="*80)
        sig_pairs = []
        for i in range(len(df.columns)):
            for j in range(i+1, len(df.columns)):
                p_val = nemenyi_results.iloc[i, j]
                if p_val < 0.05:
                    method1 = df.columns[i]
                    method2 = df.columns[j]
                    rank1 = summary[summary['Method'] == method1]['Average_Rank'].values[0]
                    rank2 = summary[summary['Method'] == method2]['Average_Rank'].values[0]
                    sig_pairs.append({
                        'Method_1': method1,
                        'Method_2': method2,
                        'p_value': p_val,
                        'Rank_1': rank1,
                        'Rank_2': rank2,
                        'Better': method1 if rank1 < rank2 else method2
                    })
        if sig_pairs:
            sig_df = pd.DataFrame(sig_pairs).sort_values('p_value')
            print(sig_df.to_string(index=False))
            print(f"\n✓ Found {len(sig_pairs)} significantly different pairs")
        else:
            print("No significantly different pairs found.")
            sig_df = None
        return nemenyi_results, sig_df
    except ImportError:
        print("\n⚠ scikit-posthocs not installed. Skipping Nemenyi test.")
        print("  Install with: pip install scikit-posthocs")
        return None, None
    except Exception as e:
        print(f"\n✗ Error in Nemenyi test: {e}")
        return None, None

def phase11_friedman_test():
    """Phase 11: Friedman statistical test."""
    print("\n" + "="*80)
    print("PHASE 11: FRIEDMAN TEST")
    print("="*80 + "\n")
    print("Step 1: Collecting results from all phases...")
    results = collect_all_results()
    if not results:
        print("\n✗ No results found! Please run Phases 1-10 first.")
        return
    print(f"\n✓ Collected results from {len(results)} methods")
    print("\nStep 2: Creating accuracy matrix...")
    df = create_accuracy_matrix(results)
    if df.shape[0] < 2 or df.shape[1] < 2:
        print("\n✗ Insufficient data for Friedman test!")
        print(f"   Need at least 2 datasets and 2 methods.")
        print(f"   Found: {df.shape[0]} datasets, {df.shape[1]} methods")
        return
    print("\nStep 3: Performing Friedman test...")
    summary, statistic, p_value = perform_friedman_test(df)
    print("\nStep 4: Saving results...")
    output_file = save_friedman_results(df, summary, statistic, p_value)
    if p_value < 0.05:
        print("\nStep 5: Performing post-hoc Nemenyi test...")
        nemenyi_results, sig_pairs = perform_posthoc_nemenyi(df, summary)
        if nemenyi_results is not None:
            with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
                nemenyi_results.to_excel(writer, sheet_name="Nemenyi_pvalues")
                if sig_pairs is not None:
                    sig_pairs.to_excel(writer, sheet_name="Significant_Pairs", index=False)
            print(f"\n✓ Nemenyi results added to: {output_file}")
    else:
        print("\nStep 5: Skipping post-hoc test (Friedman not significant)")
    print("\n" + "="*80)
    print("PHASE 11 COMPLETE!")
    print("="*80)
    print(f"\n📊 Results file: {output_file}")
    print(f"\nTop 5 methods by rank:")
    print(summary.head(5)[['Method', 'Average_Rank', 'Mean_Accuracy']].to_string(index=False))
    print("\n" + "="*80 + "\n")
    return output_file

# ============================================================================
# PHASE 12: WILCOXON PAIRWISE TESTS
# ============================================================================

def wilcoxon_safe(x, y):
    """stats.wilcoxon, but tolerant of pairs that never differ.

    Two methods can score identically on every dataset (small or easy data,
    or two variants that reduce to the same neighbourhood).  x - y is then all
    zeros, the signed-rank test has nothing to rank and SciPy raises.  That is
    not an error in the run: it means "no measurable difference", so report
    statistic 0 with p = 1.0 and let the phase carry on with the other pairs.
    """
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if not np.any(diff[~np.isnan(diff)]):
        return 0.0, 1.0
    return stats.wilcoxon(x, y, alternative='two-sided')


def phase12_wilcoxon_pairwise_tests():
    """Phase 12: Pairwise Wilcoxon signed-rank tests."""
    print("\n" + "="*80)
    print("PHASE 12: WILCOXON PAIRWISE TESTS")
    print("="*80 + "\n")
    friedman_file = get_latest_file('results_phase11_friedman_')
    if not friedman_file:
        print("✗ No Friedman results found! Run Phase 11 first.")
        return
    print(f"📂 Reading Friedman results: {friedman_file}")
    df = pd.read_excel(friedman_file, sheet_name="Accuracy_Matrix", index_col=0)
    dataset_names = df.index.tolist()
    method_names = df.columns.tolist()
    data = df.values
    print(f"\nNumber of datasets: {data.shape[0]}")
    print(f"Number of methods: {data.shape[1]}")
    print(f"Total pairwise comparisons: {data.shape[1] * (data.shape[1] - 1) // 2}")
    print("\nStep 1: Performing pairwise Wilcoxon signed-rank tests...")
    results = []
    for i, j in combinations(range(data.shape[1]), 2):
        method1 = method_names[i]
        method2 = method_names[j]
        statistic, p_value = wilcoxon_safe(data[:, i], data[:, j])
        mean_diff = data[:, i].mean() - data[:, j].mean()
        better_method = ('Tie' if mean_diff == 0
                         else method1 if mean_diff > 0 else method2)
        results.append({
            'Method_1': method1,
            'Method_2': method2,
            'Statistic': statistic,
            'P_Value': p_value,
            'Mean_Diff': mean_diff,
            'Better_Method': better_method,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values('P_Value')
    print("\n" + "="*80)
    print("PAIRWISE WILCOXON SIGNED-RANK TEST RESULTS")
    print("="*80)
    print(f"\nTotal comparisons: {len(results_df)}")
    print(f"Significant differences (p < 0.05): {(results_df['P_Value'] < 0.05).sum()}")
    print("\n" + "="*80)
    print("TOP 20 MOST SIGNIFICANT DIFFERENCES")
    print("="*80)
    print(results_df_sorted.head(20).to_string(index=False))
    bonferroni_alpha = 0.05 / len(results_df)
    print(f"\n" + "="*80)
    print(f"BONFERRONI CORRECTION")
    print("="*80)
    print(f"Original alpha: 0.05")
    print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.6f}")
    print(f"Significant with Bonferroni: {(results_df['P_Value'] < bonferroni_alpha).sum()}")
    bonferroni_sig = results_df[results_df['P_Value'] < bonferroni_alpha].sort_values('P_Value')
    if len(bonferroni_sig) > 0:
        print("\nSignificant differences after Bonferroni correction:")
        print(bonferroni_sig.to_string(index=False))
    else:
        print("\nNo significant differences after Bonferroni correction.")
    print("\n" + "="*80)
    print("STEP 2: MODEL TYPE COMPARISONS (CustomKNN vs MutualKNN vs SymmetricKNN)")
    print("="*80)
    model_comparisons = compare_model_types(results_df, method_names)
    print("\nStep 3: Creating significance matrix...")
    sig_matrix = np.zeros((len(method_names), len(method_names)))
    for _, row in results_df.iterrows():
        i = method_names.index(row['Method_1'])
        j = method_names.index(row['Method_2'])
        sig_matrix[i, j] = 1 if row['P_Value'] < 0.05 else 0
        sig_matrix[j, i] = sig_matrix[i, j]
    sig_matrix_df = pd.DataFrame(sig_matrix, index=method_names, columns=method_names)
    timestamp = datetime.now().strftime("%y_%m_%d_%H%M%S")
    output_file = results_path(f"results_phase12_wilcoxon_{timestamp}.xlsx")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        results_df_sorted.to_excel(writer, sheet_name='All_Comparisons', index=False)
        if len(bonferroni_sig) > 0:
            bonferroni_sig.to_excel(writer, sheet_name='Bonferroni_Significant', index=False)
        sig_matrix_df.to_excel(writer, sheet_name='Significance_Matrix')
        for comparison_name, comparison_df in model_comparisons.items():
            sheet_name = comparison_name[:31]
            comparison_df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    print("\n" + "="*80)
    print("PHASE 12 COMPLETE!")
    print("="*80 + "\n")
    return output_file

def compare_model_types(results_df, method_names):
    """Compare CustomKNN vs MutualKNN vs SymmetricKNN for each method variant."""
    comparisons = {}
    method_configs = set()
    for method in method_names:
        parts = method.split('_', 1)
        if len(parts) == 2:
            method_configs.add(parts[1])
    print("\nFound configurations:", sorted(method_configs))
    for config in sorted(method_configs):
        custom_method = f"CustomKNN_{config}"
        mutual_method = f"MutualKNN_{config}"
        symmetric_method = f"SymmetricKNN_{config}"
        if (custom_method in method_names and
                mutual_method in method_names and
                symmetric_method in method_names):
            print(f"\n{'='*80}")
            print(f"CONFIGURATION: {config}")
            print('='*80)
            config_comparisons = []
            for m1, m2, label in [
                (custom_method,   mutual_method,    'CustomKNN vs MutualKNN'),
                (custom_method,   symmetric_method, 'CustomKNN vs SymmetricKNN'),
                (mutual_method,   symmetric_method, 'MutualKNN vs SymmetricKNN'),
            ]:
                cmp = results_df[
                    ((results_df['Method_1'] == m1) & (results_df['Method_2'] == m2)) |
                    ((results_df['Method_1'] == m2) & (results_df['Method_2'] == m1))
                ]
                if not cmp.empty:
                    row = cmp.iloc[0]
                    stars = ('***' if row['P_Value'] < 0.001 else
                             '**'  if row['P_Value'] < 0.01  else
                             '*'   if row['P_Value'] < 0.05  else 'ns')
                    config_comparisons.append({
                        'Comparison': label,
                        'P_Value': row['P_Value'],
                        'Mean_Diff': abs(row['Mean_Diff']),
                        'Better': row['Better_Method'].split('_')[0],
                        'Significant': row['Significant']
                    })
                    print(f"  {label}: p={row['P_Value']:.6f} {stars} | Better: {row['Better_Method']}")
            if config_comparisons:
                comparisons[f"Compare_{config}"] = pd.DataFrame(config_comparisons)
    print(f"\n{'='*80}")
    print("SUMMARY: MODEL TYPE WINS ACROSS ALL CONFIGURATIONS")
    print('='*80)
    custom_wins = mutual_wins = symmetric_wins = 0
    for config_df in comparisons.values():
        for _, row in config_df.iterrows():
            if row['Significant'] == 'Yes':
                if row['Better'] == 'CustomKNN':
                    custom_wins += 1
                elif row['Better'] == 'MutualKNN':
                    mutual_wins += 1
                elif row['Better'] == 'SymmetricKNN':
                    symmetric_wins += 1
    print(f"\nSignificant wins (p < 0.05):")
    print(f"  CustomKNN:    {custom_wins}")
    print(f"  MutualKNN:    {mutual_wins}")
    print(f"  SymmetricKNN: {symmetric_wins}")
    comparisons['Summary_Model_Wins'] = pd.DataFrame({
        'Model': ['CustomKNN', 'MutualKNN', 'SymmetricKNN'],
        'Significant_Wins': [custom_wins, mutual_wins, symmetric_wins]
    })
    return comparisons

# ============================================================================
# PHASE 13: COMBINED CPU TIME REPORT
# ============================================================================

def _get_latest_csv(pattern):
    """Return the most recent CSV file matching a prefix pattern."""
    if not os.path.isdir(RESULTS_FOLDER):
        return None
    files = [f for f in os.listdir(RESULTS_FOLDER)
             if f.startswith(pattern) and f.endswith('.csv')]
    return os.path.join(RESULTS_FOLDER, sorted(files, reverse=True)[0]) if files else None


def _find_time_column(df):
    """Find the timing column — handles both 'CPU Time (s)' and 'GPU Time (s)'."""
    for col in ["CPU Time (s)", "GPU Time (s)"]:
        if col in df.columns:
            return col
    raise KeyError("No timing column found (expected 'CPU Time (s)' or 'GPU Time (s)')")


def _load_phase2_cpu(file):
    """Dataset | Model | Best_k | CV_Accuracy | [CPU/GPU] Time (s)  — from Phase 2 Excel."""
    df = pd.read_excel(file, sheet_name="Best_k_Summary")
    tc = _find_time_column(df)
    df = df[["Dataset", "Model", tc]].rename(columns={tc: "Phase2_CPU"})
    # Phase 2 is the tuning cost: the whole grid, over every fold.
    return df


def _load_phase3_cpu(file):
    """Dataset | Model | mean per-fold time — from Phase 3 Excel."""
    df = pd.read_excel(file, sheet_name="Summary")
    keep = [c for c in ["Dataset", "Model"] if c in df.columns]
    df = df[keep + ["Time_Mean"]].rename(columns={"Time_Mean": "Phase3_CPU"})
    return df


def _load_phase5_cpu(file):
    """Dataset | Model | k_max | [CPU/GPU] Time (s)  — from Phase 5 CSV."""
    df = pd.read_csv(file)
    tc = _find_time_column(df)
    df = df[["Dataset", "Model", "k_max", tc]].rename(columns={tc: "Phase5_CPU"})
    return df


def _load_phase6_cpu(file):
    """Dataset | Model | k_max | mean per-fold time — from Phase 6 Excel."""
    df = pd.read_excel(file, sheet_name="Summary")
    keep = [c for c in ["Dataset", "Model", "k_max"] if c in df.columns]
    df = df[keep + ["Time_Mean"]].rename(columns={"Time_Mean": "Phase6_CPU"})
    return df


def _load_phase7_cpu(file):
    """Dataset | Model | k_max | [CPU/GPU] Time (s)  — from Phase 7 CSV."""
    df = pd.read_csv(file)
    tc = _find_time_column(df)
    df = df[["Dataset", "Model", "k_max", tc]].rename(columns={tc: "Phase7_CPU"})
    return df


def _load_phase8_cpu(file):
    """Dataset | Model | k_max | mean per-fold time — from Phase 8 Excel."""
    df = pd.read_excel(file, sheet_name="Summary")
    keep = [c for c in ["Dataset", "Model", "k_max"] if c in df.columns]
    df = df[keep + ["Time_Mean"]].rename(columns={"Time_Mean": "Phase8_CPU"})
    return df


def _load_phase9_cpu(file):
    """Dataset | Model | k_max | window | [CPU/GPU] Time (s)  — from Phase 9 CSV."""
    df = pd.read_csv(file)
    tc = _find_time_column(df)
    if "window" not in df.columns:
        df["window"] = STABILITY_WINDOW
    df = df[["Dataset", "Model", "k_max", "window", tc]].rename(columns={tc: "Phase9_CPU"})
    return df


def _load_phase10_cpu(file):
    """Dataset | Model | k_max | window | mean per-fold time — from Phase 10 Excel."""
    df = pd.read_excel(file, sheet_name="Summary")
    keep = [c for c in ["Dataset", "Model", "k_max", "window"] if c in df.columns]
    if "window" not in keep:
        df["window"] = STABILITY_WINDOW
        keep.append("window")
    df = df[keep + ["Time_Mean"]].rename(columns={"Time_Mean": "Phase10_CPU"})
    return df


def phase13_combined_cpu_report():
    """Phase 13: Build an Excel with four sheets combining CPU times across phase pairs."""

    print("\n" + "="*80)
    print("PHASE 13: COMBINED CPU TIME REPORT")
    print("="*80 + "\n")

    timestamp = datetime.now().strftime("%y_%m_%d_%H%M%S")
    output_file = results_path(f"results_phase13_cpu_combined_{timestamp}.xlsx")

    phase2_file = get_latest_file("results_phase2_best_k_")
    phase3_file = get_latest_file("results_phase3_best_k_test_")
    phase5_file = _get_latest_csv("results_phase5_adaptive_timing_")
    phase6_file = get_latest_file("results_phase6_adaptive_test_")
    phase7_file = _get_latest_csv("results_phase7_sqrt_timing_")
    phase8_file = get_latest_file("results_phase8_sqrt_test_")
    phase9_file = _get_latest_csv("results_phase9_last_k_timing_")
    phase10_file = get_latest_file("results_phase10_last_k_test_")

    sheets = {}

    # ------------------------------------------------------------------
    # Sheet 0 — Phase 2 + Phase 3
    # Join key: Dataset, Model
    # ------------------------------------------------------------------
    if phase2_file and phase3_file:
        print(f"📂 Phase 2 : {phase2_file}")
        print(f"📂 Phase 3 : {phase3_file}")
        p2 = _load_phase2_cpu(phase2_file)
        p3 = _load_phase3_cpu(phase3_file)
        merged = pd.merge(p2, p3, on=["Dataset", "Model"], how="outer")
        merged["Total_CPU"] = merged["Phase2_CPU"].fillna(0) + merged["Phase3_CPU"].fillna(0)
        merged = merged.sort_values(["Dataset", "Model"]).reset_index(drop=True)
        sheets["Phase2_plus_Phase3"] = merged
        print(f"  ✓ Phase2+Phase3: {len(merged)} rows")
    else:
        print(f"  ⚠ Skipping Phase2+Phase3 — missing files "
              f"(phase2={'found' if phase2_file else 'MISSING'}, "
              f"phase3={'found' if phase3_file else 'MISSING'})")

    # ------------------------------------------------------------------
    # Sheet 1 — Phase 3 + Phase 5
    # Join key: Dataset, Model   (Phase 5 has one row per k_max, keep all)
    # ------------------------------------------------------------------
    if phase3_file and phase5_file:
        print(f"📂 Phase 3 : {phase3_file}")
        print(f"📂 Phase 5 : {phase5_file}")
        p3 = _load_phase3_cpu(phase3_file)
        p4 = _load_phase5_cpu(phase5_file)
        merged = pd.merge(p3, p4, on=["Dataset", "Model"], how="outer")
        merged["Total_CPU"] = merged["Phase3_CPU"].fillna(0) + merged["Phase5_CPU"].fillna(0)
        merged = merged.sort_values(["Dataset", "Model", "k_max"]).reset_index(drop=True)
        sheets["Phase3_plus_Phase5"] = merged
        print(f"  ✓ Phase3+Phase5: {len(merged)} rows")
    else:
        print(f"  ⚠ Skipping Phase3+Phase5 — missing files "
              f"(phase3={'found' if phase3_file else 'MISSING'}, "
              f"phase5={'found' if phase5_file else 'MISSING'})")

    # ------------------------------------------------------------------
    # Sheet 2 — Phase 5 + Phase 6
    # Join key: Dataset, Model, k_max
    # ------------------------------------------------------------------
    if phase5_file and phase6_file:
        print(f"\n📂 Phase 5 : {phase5_file}")
        print(f"📂 Phase 6 : {phase6_file}")
        p4 = _load_phase5_cpu(phase5_file)
        p5 = _load_phase6_cpu(phase6_file)
        merged = pd.merge(p4, p5, on=["Dataset", "Model", "k_max"], how="outer")
        merged["Total_CPU"] = merged["Phase5_CPU"].fillna(0) + merged["Phase6_CPU"].fillna(0)
        merged = merged.sort_values(["Dataset", "Model", "k_max"]).reset_index(drop=True)
        sheets["Phase5_plus_Phase6"] = merged
        print(f"  ✓ Phase5+Phase6: {len(merged)} rows")
    else:
        print(f"  ⚠ Skipping Phase5+Phase6 — missing files "
              f"(phase5={'found' if phase5_file else 'MISSING'}, "
              f"phase6={'found' if phase6_file else 'MISSING'})")

    # ------------------------------------------------------------------
    # Sheet 3 — Phase 7 + Phase 8
    # Join key: Dataset, Model
    # ------------------------------------------------------------------
    if phase7_file and phase8_file:
        print(f"\n📂 Phase 7 : {phase7_file}")
        print(f"📂 Phase 8 : {phase8_file}")
        p6 = _load_phase7_cpu(phase7_file)
        p7 = _load_phase8_cpu(phase8_file)
        merged = pd.merge(p6, p7, on=["Dataset", "Model"], how="outer")
        merged["Total_CPU"] = merged["Phase7_CPU"].fillna(0) + merged["Phase8_CPU"].fillna(0)
        merged = merged.sort_values(["Dataset", "Model"]).reset_index(drop=True)
        sheets["Phase7_plus_Phase8"] = merged
        print(f"  ✓ Phase7+Phase8: {len(merged)} rows")
    else:
        print(f"  ⚠ Skipping Phase7+Phase8 — missing files "
              f"(phase7={'found' if phase7_file else 'MISSING'}, "
              f"phase8={'found' if phase8_file else 'MISSING'})")

    # ------------------------------------------------------------------
    # Sheet 4 — Phase 9 + Phase 10
    # Join key: Dataset, Model   (window is fixed; k_max is fixed)
    # ------------------------------------------------------------------
    if phase9_file and phase10_file:
        print(f"\n📂 Phase 9 : {phase9_file}")
        print(f"📂 Phase 10 : {phase10_file}")
        p8 = _load_phase9_cpu(phase9_file)
        p9 = _load_phase10_cpu(phase10_file)
        merged = pd.merge(p8, p9, on=["Dataset", "Model", "k_max", "window"], how="outer")
        merged["Total_CPU"] = merged["Phase9_CPU"].fillna(0) + merged["Phase10_CPU"].fillna(0)
        merged = merged.sort_values(["Dataset", "Model"]).reset_index(drop=True)
        sheets["Phase9_plus_Phase10"] = merged
        print(f"  ✓ Phase9+Phase10: {len(merged)} rows")
    else:
        print(f"  ⚠ Skipping Phase9+Phase10 — missing files "
              f"(phase9={'found' if phase9_file else 'MISSING'}, "
              f"phase10={'found' if phase10_file else 'MISSING'})")

    if not sheets:
        print("\n✗ No data available to write. Run earlier phases first.")
        return None

    # ------------------------------------------------------------------
    # Phase 4 — nested CV. Not a pair: it is one self-contained measurement
    # per (dataset, method), so it gets its own sheet with the per-fold cost
    # and the total the whole phase spent on that cell.
    # ------------------------------------------------------------------
    phase4_file = get_latest_file("results_phase4_nested_cv_")
    if phase4_file:
        print(f"📂 Phase 4 : {phase4_file}")
        try:
            p4 = pd.read_excel(phase4_file, sheet_name="Summary")
            keep = ["Dataset", "Method", "Model", "Folds",
                    "Time_Mean", "Time_Std", "Time_Total", "Accuracy_Mean"]
            p4 = p4[[c for c in keep if c in p4.columns]].copy()
            p4 = p4.rename(columns={"Time_Mean": "Phase4_CPU_per_fold",
                                    "Time_Std": "Phase4_CPU_std",
                                    "Time_Total": "Phase4_CPU_total"})
            sheets["Phase4_NestedCV"] = p4.sort_values(["Dataset", "Method"])
            print(f"  ✓ Phase4 nested CV: {len(p4)} rows")
        except Exception as exc:
            print(f"  ⚠ Skipping Phase 4 timing — {exc}")
    else:
        print("  ⚠ Skipping Phase 4 timing — no nested-CV results")

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n✓ Phase 13 complete! Results: {output_file}")
    print(f"  Sheets written: {list(sheets.keys())}")
    print("="*80 + "\n")
    return output_file


# ============================================================================
# PER-DATASET RESULTS
#   One stand-alone workbook per dataset, gathered from whichever phase files
#   exist.  Safe to run at any time — it only reads the phase results.
# ============================================================================

# phase → (file prefix, is_csv, how to name the configuration of a row)
_PER_DATASET_SOURCES = [
    (1, "results_phase1_initial_",     lambda r: f"k{int(r['k'])}"),
    (3, "results_phase3_best_k_test_", lambda r: "BestK"),
    (6, "results_phase6_adaptive_test_", lambda r: f"Adaptive_k{int(r['k_max'])}"),
    (8, "results_phase8_sqrt_test_",   lambda r: "Sqrt_n2"),
    (10, "results_phase10_last_k_test_", lambda r: f"LastK_w{int(r['window'])}"),
]

_SUMMARY_COLUMNS = ["Phase", "Configuration", "Model", "k", "k_max", "window",
                    "Time (s)", "Accuracy", "Precision", "Recall", "F1",
                    "k_mean", "k_std", "k_min", "k_max_used"]


def _sheet_dataset(sheet_name, datasets):
    """Which dataset a per-class report sheet belongs to (longest name wins)."""
    for name in sorted(datasets, key=len, reverse=True):
        if sheet_name.startswith(name + "_"):
            return name
    return None


def export_per_dataset_results():
    """Write results/per_dataset/<dataset>.xlsx for every dataset with results."""
    print("\n" + "="*80)
    print("PER-DATASET RESULTS")
    print("="*80 + "\n")

    summary = {}        # dataset → list of metric rows
    reports = {}        # dataset → list of per-class report rows
    best_k = {}
    cv_scores = {}
    build_timing = {}

    def bucket(store, dataset):
        return store.setdefault(dataset, [])

    # ---- metrics + per-class reports from the test phases ----
    for phase, prefix, name_config in _PER_DATASET_SOURCES:
        path = get_latest_file(prefix)
        if not path:
            continue
        print(f"📂 Phase {phase}: {os.path.basename(path)}")
        try:
            metrics = pd.read_excel(path, sheet_name="Summary")
        except Exception as e:
            print(f"  ⚠ unreadable: {e}")
            continue
        tcol = "Time_Mean" if "Time_Mean" in metrics.columns else None
        # Summary carries each metric twice: the number in <metric>_Mean and the
        # "m ± s" text in <metric>.  Drop the text before renaming, or the two
        # collide into a duplicate column and every later lookup returns a frame.
        named = ("Accuracy", "Precision", "Recall", "F1")
        metrics = metrics.drop(columns=[m for m in named if m in metrics.columns])
        metrics = metrics.rename(columns={f"{m}_Mean": m for m in named
                                          if f"{m}_Mean" in metrics.columns})
        for _, row in metrics.iterrows():
            entry = {"Phase": phase, "Configuration": name_config(row), "Model": row["Model"],
                     "Time (s)": row[tcol] if tcol else None}
            for col in ("k", "k_max", "window", "Accuracy", "Precision", "Recall", "F1",
                        "k_mean", "k_std", "k_min", "k_max_used"):
                if col in metrics.columns:
                    entry[col] = row[col]
            bucket(summary, row["Dataset"]).append(entry)

        # the classification-report sheets that sit next to "Metrics"
        known = set(metrics["Dataset"].astype(str))
        try:
            with pd.ExcelFile(path) as xls:
                for sheet in xls.sheet_names:
                    if sheet == "Metrics":
                        continue
                    dataset = _sheet_dataset(sheet, known)
                    if dataset is None:
                        continue
                    report = pd.read_excel(xls, sheet_name=sheet, index_col=0)
                    report = report.reset_index().rename(columns={report.index.name or "index": "Class"})
                    report.insert(0, "Source", sheet)
                    report.insert(0, "Phase", phase)
                    bucket(reports, dataset).append(report)
        except Exception as e:
            print(f"  ⚠ could not read the report sheets: {e}")

    # ---- phase 2: best k and the full CV curve ----
    path = get_latest_file("results_phase2_best_k_")
    if path:
        print(f"📂 Phase 2: {os.path.basename(path)}")
        try:
            df = pd.read_excel(path, sheet_name="Best_k_Summary")
            for dataset, part in df.groupby("Dataset"):
                best_k[dataset] = part.drop(columns=["Dataset"]).reset_index(drop=True)
            scores = pd.read_excel(path, sheet_name="All_CV_Scores")
            for dataset, part in scores.groupby("Dataset"):
                cv_scores[dataset] = part.pivot_table(index="k", columns="Model",
                                                      values="CV_Accuracy")
        except Exception as e:
            print(f"  ⚠ unreadable: {e}")

    # ---- phases 4/6/8: dataset-building times ----
    for phase, prefix in ((4, "results_phase5_adaptive_timing_"),
                          (6, "results_phase7_sqrt_timing_"),
                          (8, "results_phase9_last_k_timing_")):
        path = _get_latest_csv(prefix)
        if not path:
            continue
        print(f"📂 Phase {phase}: {os.path.basename(path)}")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  ⚠ unreadable: {e}")
            continue
        df.insert(0, "Phase", phase)
        for dataset, part in df.groupby("Dataset"):
            bucket(build_timing, dataset).append(part.drop(columns=["Dataset"]))

    datasets = sorted(set(summary) | set(best_k) | set(cv_scores) | set(build_timing))
    if not datasets:
        print("\n✗ No result files to split — run at least one test phase first.\n")
        return None

    os.makedirs(PER_DATASET_FOLDER, exist_ok=True)
    print()
    for dataset in datasets:
        out_file = os.path.join(PER_DATASET_FOLDER, f"{dataset}.xlsx")
        try:
            with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
                rows = summary.get(dataset)
                if rows:
                    table = pd.DataFrame(rows)
                    for col in _SUMMARY_COLUMNS:
                        if col not in table.columns:
                            table[col] = np.nan
                    table = table[_SUMMARY_COLUMNS].sort_values(
                        "Accuracy", ascending=False).reset_index(drop=True)
                    table.insert(0, "Rank", range(1, len(table) + 1))
                    table = table.dropna(axis=1, how="all")
                    table.to_excel(writer, sheet_name="Summary", index=False)
                if dataset in best_k:
                    best_k[dataset].to_excel(writer, sheet_name="Best_k", index=False)
                if dataset in cv_scores:
                    cv_scores[dataset].to_excel(writer, sheet_name="CV_Scores")
                if dataset in build_timing:
                    pd.concat(build_timing[dataset], ignore_index=True).to_excel(
                        writer, sheet_name="Build_Timing", index=False)
                if dataset in reports:
                    pd.concat(reports[dataset], ignore_index=True).to_excel(
                        writer, sheet_name="Class_Reports", index=False)
            n_methods = len(summary.get(dataset, []))
            print(f"✓ {dataset:20s} → {out_file}  ({n_methods} method(s))")
        except Exception as e:
            print(f"✗ {dataset:20s} → {type(e).__name__}: {e}")

    print(f"\n✓ Per-dataset results written to {PER_DATASET_FOLDER}/\n")
    return PER_DATASET_FOLDER


# ============================================================================
# MAIN
# ============================================================================

Variant = namedtuple("Variant", "tag label base_folder variant_dir results_folder")


def build_variants():
    """
    The dataset collections a run covers.

    Clean keeps the original layout (base_folder/, adaptive_k*/, results/);
    every noisy variant gets its own sub-folders named after its noise setting,
    so clean and noisy results can never overwrite each other.
    """
    variants = []
    if NOISE_RUN_CLEAN or not NOISE_ENABLED:
        variants.append(Variant("clean", "clean data", CLEAN_BASE_FOLDER, "", OUTPUT_FOLDER))
    if NOISE_ENABLED:
        for rate in NOISE_RATES:
            tag = noise_tag(rate)
            # Read the CLEAN pool. Every phase now injects noise into the
            # TRAINING rows of each fold via fold_labels(), so pointing the
            # variant at the pre-noised folder would corrupt the labels twice
            # AND leave corrupted rows in the held-out folds, which are meant
            # to be the clean ground truth. Phase 4 always did this for itself;
            # the CV rewrite made it the right behaviour for every phase.
            variants.append(Variant(
                tag, f"{rate*100:g}% label noise (seed {NOISE_SEED})",
                CLEAN_BASE_FOLDER, tag, os.path.join(OUTPUT_FOLDER, tag)))
    return variants


def activate_variant(variant):
    """Point the pipeline at one dataset collection."""
    global BASE_FOLDER, VARIANT_DIR, RESULTS_FOLDER, PER_DATASET_FOLDER, ACTIVE_NOISE_RATE
    BASE_FOLDER = variant.base_folder
    VARIANT_DIR = variant.variant_dir
    RESULTS_FOLDER = variant.results_folder
    PER_DATASET_FOLDER = os.path.join(RESULTS_FOLDER, "per_dataset")
    ACTIVE_NOISE_RATE = next((r for r in NOISE_RATES if variant.tag == noise_tag(r)),
                             None) if NOISE_ENABLED else None


# ============================================================================
# PHASE 4: NESTED (DOUBLE) CROSS-VALIDATION
#
#   Phases 2-3 and 5-10 report ONE number per (dataset, method): the score on
#   data_te1 split.  On 150 rows (iris) or 336 (ecoli) that single number
#   carries several points of split-to-split noise, and the Friedman / Wilcoxon
#   the fixed split, and the tests in Phases 11-12 inherit every bit of it.
#
#   This phase re-estimates the same methods with the protocol reviewers ask
#   for — repeated stratified k-fold — while keeping the k-selection honest:
#
#       outer  RepeatedStratifiedKFold(outer_folds x repeats)  → MEASURES
#         inner  StratifiedKFold(inner_folds) on the outer-TRAIN → TUNES k
#
#   Tuning inside every outer fold is what makes it "double" / nested CV.  The
#   reason it cannot be collapsed into a single CV loop: Phase 2 already spends
#   cross-validation on choosing k.  Scoring on those same folds would let the
#   held-out labels influence the model that is then graded on them, and the
#   reported accuracy comes out optimistically biased.
#
#   data1 and data_te1 are concatenated first, so the outer CV resamples the
#   WHOLE dataset and no row is permanently a test row.  Answering the reviewer
#   with CV over the 70 % training part only would not address the objection.
#
#   The adaptive-k methods (Phases 5-10) are recomputed from scratch inside each
#   outer fold.  Their per-row k comes from leave-one-out over the training
#   set, so a k computed once over all the data would have already seen the
#   outer test rows.  Recomputing is what makes this phase expensive: budget
#   roughly outer_folds x repeats times the cost of Phases 5-10.
#
#   Label noise, when a noisy variant is active, is injected into the outer
#   TRAINING labels only, fold by fold, starting from the clean data — the same
#   rule the file-based noisy copies follow (test labels are never touched).
# ============================================================================

MODEL_MAP = {'CustomKNN': CustomKNN, 'MutualKNN': MutualKNN, 'SymmetricKNN': SymmetricKNN}


def nested_method_specs():
    """(display name, model, kind, parameter) for every method the phase scores."""
    specs = []
    for model in MODELS:
        if "fixed" in NESTED_METHODS:
            for k in NESTED_FIXED_K:
                specs.append((f"{model}_k{k}", model, "fixed", int(k)))
        if "bestk" in NESTED_METHODS:
            specs.append((f"{model}_BestK", model, "bestk", None))
        if "adaptive" in NESTED_METHODS:
            for k_max in ADAPTIVE_K_MAX_VALUES:
                specs.append((f"{model}_Adaptive_kmax{k_max}", model, "adaptive", int(k_max)))
        if "sqrt" in NESTED_METHODS:
            specs.append((f"{model}_Sqrt", model, "sqrt", None))
        if "lastk" in NESTED_METHODS:
            specs.append((f"{model}_LastK_w{STABILITY_WINDOW}", model, "lastk", None))
        if "lastk_tuned" in NESTED_METHODS:
            specs.append((f"{model}_LastK_tunedW", model, "lastk_tuned", None))
    return specs


def load_pooled_dataset(dataset_path):
    """
    data1 + data_te1 concatenated — the pool the outer CV resamples.

    The fixed split stops being a property of the data and becomes just one of
    the many splits the outer loop draws.
    """
    frames = []
    for name in ("data1", "data_te1"):
        path = os.path.join(dataset_path, name)
        if os.path.exists(path):
            frames.append(pd.read_csv(path, sep="\t", header=None))
    if not frames:
        return None, None
    data = pd.concat(frames, ignore_index=True)
    return data.iloc[:, :-1].values, data.iloc[:, -1].values


def stratified_plan(y, requested, policy):
    """
    How many folds stratification can actually honour, and on which rows.

    A class with fewer members than folds cannot be present in every fold —
    ecoli has classes with 2 instances, so 10-fold cannot represent them.
    Returns (n_splits, keep_mask, note); the note is printed and stored so the
    compromise ends up in the paper rather than in a silent sklearn warning.

      "warn"   — keep the requested folds, name the classes that cannot fill them
      "reduce" — drop to the largest fold count every class can fill
      "drop"   — remove the under-populated classes from the run
    """
    y = np.asarray(y)
    counts = Counter(y.tolist())
    keep = np.ones(len(y), dtype=bool)
    if not counts:
        return 0, keep, "empty dataset"
    smallest = min(counts.values())
    rare = {c: n for c, n in counts.items() if n < requested}
    if not rare:
        return requested, keep, ""

    rare_txt = ", ".join(f"{c} (n={n})" for c, n in sorted(rare.items(), key=lambda kv: str(kv[0])))
    if policy == "reduce":
        n_splits = max(2, min(requested, smallest))
        return n_splits, keep, (f"folds reduced {requested}→{n_splits}; "
                                f"smallest class has {smallest} members")
    if policy == "drop":
        keep = np.array([c not in rare for c in y.tolist()])
        remaining = Counter(y[keep].tolist())
        if len(remaining) < 2:
            return 0, keep, f"dropping rare classes leaves fewer than 2 classes ({rare_txt})"
        n_splits = max(2, min(requested, min(remaining.values())))
        return n_splits, keep, (f"dropped {int((~keep).sum())} rows from rare classes: {rare_txt}")
    return requested, keep, (f"{rare_txt} — fewer members than {requested} folds, "
                             f"so these classes cannot appear in every fold")


def tune_k_inner(model_class, X, y, k_range, inner_folds, seed):
    """
    Inner loop: pick k by stratified CV on the OUTER-TRAINING data alone.

    Same selection rule as Phase 2 (mean CV accuracy, CV_TIE_BREAKING breaks
    ties) so the two protocols stay comparable — only the data it sees and the
    stratification differ.
    """
    n_splits, _, _ = stratified_plan(y, inner_folds, "reduce")
    n_splits = max(2, min(n_splits or 2, len(y)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))

    max_k = min(len(tr) for tr, _ in splits)
    usable = [int(k) for k in k_range if 1 <= k <= max_k] or [1]

    mean_scores = {}
    for k in usable:
        scores = []
        for tr, va in splits:
            model = model_class(k=k)
            model.fit(X[tr], y[tr])
            scores.append(accuracy_score(y[va], model.predict(X[va])))
        mean_scores[k] = float(np.mean(scores))

    best_score = max(mean_scores.values())
    tied = [k for k, s in mean_scores.items() if s == best_score]
    best_k = max(tied) if CV_TIE_BREAKING == "highest_k" else min(tied)
    return best_k, best_score, mean_scores


def tune_window_inner(model_name, X, y, w_values, k_max, inner_folds, seed):
    """
    Choose the stability window by cross-validation on the OUTER-TRAINING data.

    The correctness matrix does not depend on w, so each inner fold builds it
    once and every candidate window is scored by re-scanning it — the sweep's
    trick, reused here so that honest selection costs little more than a fixed
    window would.
    """
    n_splits, _, _ = stratified_plan(y, inner_folds, "reduce")
    n_splits = max(2, min(n_splits or 2, len(y)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model_class = MODEL_MAP[model_name]

    scores = {w: [] for w in w_values}
    for tr, va in skf.split(X, y):
        budget = max(1, min(int(k_max), len(tr) - 1))
        correct = compute_adaptive_k_last_stable(X[tr], y[tr], model_name, budget,
                                                 return_correctness=True)
        for w in w_values:
            if w > budget:
                continue
            predictions, _ = predict_with_adaptive_k(
                model_class, X[tr], y[tr], X[va], scan_stable_window(correct, w))
            scores[w].append(accuracy_score(y[va], predictions))

    usable = {w: float(np.mean(v)) for w, v in scores.items() if v}
    if not usable:
        return int(min(w_values)), float("nan")
    best = max(usable.values())
    tied = [w for w, value in usable.items() if value == best]
    chosen = max(tied) if CV_TIE_BREAKING == "highest_k" else min(tied)
    return int(chosen), best


def predict_with_adaptive_k(model_class, X_train, y_train, X_test, k_per_row):
    """
    Each test point inherits the k of its nearest training point.

    The same rule Phases 6/8/10 apply; kept here so a fold can run from
    arrays instead of the derived files the old protocol wrote to disk.
    """
    if get_backend() == "gpu":
        from knn_algorithms import _batched_cdist, _get_device
        dists = _batched_cdist(np.asarray(X_test, dtype=np.float32),
                               np.asarray(X_train, dtype=np.float32), _get_device())
    else:
        from scipy.spatial.distance import cdist
        dists = cdist(X_test, X_train, metric='euclidean')

    ks_used = np.asarray(k_per_row)[np.argmin(dists, axis=1)]
    predictions = np.empty(len(X_test), dtype=y_train.dtype)
    for k_val in np.unique(ks_used):
        mask = ks_used == k_val
        model = model_class(k=int(k_val))
        model.fit(X_train, y_train)
        predictions[mask] = model.predict(X_test[mask])
    return predictions, ks_used


def evaluate_nested_fold(spec, X_train, y_train, X_test, y_test, seed):
    """Score one method on one outer fold. Everything it needs it derives from X_train."""
    _, model_name, kind, param = spec
    model_class = MODEL_MAP[model_name]
    n_train = len(X_train)

    t = Timer()
    t.start()
    if kind == "fixed":
        k = max(1, min(int(param), n_train - 1))
        model = model_class(k=k)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        k_stats = {"Selected_k": k, "Selected_w": np.nan, "Inner_CV_Accuracy": np.nan,
                   "k_mean": float(k), "k_std": 0.0,
                   "k_min": k, "k_max_used": k, "k_max_budget": k}
    elif kind == "bestk":
        best_k, inner_acc, _ = tune_k_inner(model_class, X_train, y_train,
                                            CV_K_RANGE, NESTED_INNER_FOLDS, seed)
        model = model_class(k=best_k)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        k_stats = {"Selected_k": best_k, "Selected_w": np.nan,
                   "Inner_CV_Accuracy": inner_acc,
                   "k_mean": float(best_k), "k_std": 0.0,
                   "k_min": int(best_k), "k_max_used": int(best_k),
                   "k_max_budget": max(CV_K_RANGE)}
    else:
        if kind == "adaptive":
            k_max = int(param)
        elif kind == "sqrt":
            k_max = math.ceil(math.sqrt(n_train / 2))
        else:
            k_max = LAST_K_MAX
        k_max = max(1, min(int(k_max), n_train - 1))

        window = None
        if kind == "lastk_tuned":
            window, inner_acc = tune_window_inner(model_name, X_train, y_train,
                                                  NESTED_W_VALUES, k_max,
                                                  NESTED_INNER_FOLDS, seed)
            k_per_row = compute_adaptive_k_last_stable(X_train, y_train, model_name,
                                                       k_max, window)
        elif kind == "lastk":
            k_per_row = compute_adaptive_k_last_stable(X_train, y_train, model_name,
                                                       k_max, STABILITY_WINDOW)
        else:
            k_per_row = compute_first_correct_k(X_train, y_train, model_name, k_max)

        predictions, ks_used = predict_with_adaptive_k(model_class, X_train, y_train,
                                                       X_test, k_per_row)
        k_stats = {"Selected_k": np.nan,
                   "Selected_w": window if window is not None else np.nan,
                   "Inner_CV_Accuracy": inner_acc if kind == "lastk_tuned" else np.nan,
                   "k_mean": float(np.mean(ks_used)), "k_std": float(np.std(ks_used)),
                   "k_min": int(np.min(ks_used)), "k_max_used": int(np.max(ks_used)),
                   "k_max_budget": k_max}
    elapsed = t.stop()

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, average="macro", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="macro", zero_division=0),
        "F1": f1_score(y_test, predictions, average="macro", zero_division=0),
        Timer.label(): elapsed,
        **k_stats,
    }


def nested_cv_task(dataset_path, dataset_name, spec, output_file, lock, noise_rate):
    """
    Run every outer fold for ONE (dataset, method) pair.

    noise_rate is an argument rather than a global on purpose: with backend
    "gpu" the pool uses "spawn", so a worker re-imports this module and never
    sees what activate_variant() set in the parent.  Reading ACTIVE_NOISE_RATE
    here silently produced clean results for every noisy run.

    The whole pair is written in a single append at the end, so a task that
    dies leaves nothing behind and the phase resumes cleanly at pair
    granularity — the same contract as the other phases.
    """
    method_name, model_name, kind, _ = spec
    try:
        X, y = load_pooled_dataset(dataset_path)
        if X is None:
            print(f"⚠ Missing data1/data_te1: {dataset_name}")
            return None

        # The same splitter, scaler and noise the other phases use.  Phase 4 used
        # to build its own folds and skip scaling entirely, which silently made
        # its numbers incomparable with Phases 1/6 — and plainly wrong once the
        # base folder holds unscaled features.
        design = cv_design(X, y, NESTED_OUTER_FOLDS, NESTED_REPEATS, NESTED_SEED,
                           NESTED_RARE_CLASS_POLICY)
        if design is None:
            print(f"✗ {dataset_name} | {method_name}: cannot stratify")
            return None
        X, y, splits, n_splits, note = design

        rows = []
        for repeat, fold, train_idx, test_idx in splits:
            X_train, X_test = scale_fold(X[train_idx], X[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]
            y_train, n_flipped = fold_labels(y_train, dataset_name, repeat, fold, noise_rate)

            scores = evaluate_nested_fold(spec, X_train, y_train, X_test, y_test,
                                          NESTED_SEED + (repeat - 1) * n_splits + fold)
            rows.append({
                "Dataset": dataset_name, "Method": method_name, "Model": model_name,
                "Kind": kind, "Repeat": repeat, "Fold": fold,
                "N_Train": len(train_idx), "N_Test": len(test_idx),
                "Outer_Folds": n_splits, "Labels_Flipped": n_flipped,
                "Stratification_Note": note,
                **scores,
            })

        if rows:
            accs = [r["Accuracy"] for r in rows]
            safe_append_to_excel_phase4(output_file, rows, lock)
            print(f"✓ Phase 4: {dataset_name:12s} | {method_name:28s} | "
                  f"Acc={np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f} "
                  f"over {len(rows)} folds")
        return rows
    except Exception as e:
        print(f"✗ Error: {dataset_name} | {method_name} -> {e}")
        return None


def _fold_time_column(folds_df):
    """
    The timing column as the workbook was WRITTEN, not as the config reads now.

    Timer.label() follows the active timing mode, so a workbook recorded under
    "cpu" cannot be summarised while the config says "gpu" unless the column is
    looked up in the data itself.
    """
    if Timer.label() in folds_df.columns:
        return Timer.label()
    for name in ("CPU Time (s)", "GPU Time (s)"):
        if name in folds_df.columns:
            return name
    return None


def summarise_nested_folds(folds_df):
    """mean ± std over every outer fold — the table the reviewer asked for."""
    tcol = _fold_time_column(folds_df)
    if tcol is None:                       # older workbook without timings
        folds_df = folds_df.copy()
        tcol = Timer.label()
        folds_df[tcol] = np.nan
    summary = folds_df.groupby(["Dataset", "Method", "Model", "Kind"], as_index=False).agg(
        Folds=("Accuracy", "size"),
        Accuracy_Mean=("Accuracy", "mean"), Accuracy_Std=("Accuracy", "std"),
        F1_Mean=("F1", "mean"), F1_Std=("F1", "std"),
        Precision_Mean=("Precision", "mean"), Precision_Std=("Precision", "std"),
        Recall_Mean=("Recall", "mean"), Recall_Std=("Recall", "std"),
        k_Mean=("k_mean", "mean"), k_Std=("k_mean", "std"),
        Time_Mean=(tcol, "mean"), Time_Std=(tcol, "std"),
        Time_Total=(tcol, "sum"),
    )
    # Ready to paste into a results table.
    summary["Accuracy"] = [f"{m:.4f} ± {s:.4f}" for m, s in
                           zip(summary["Accuracy_Mean"], summary["Accuracy_Std"].fillna(0.0))]
    summary["F1"] = [f"{m:.4f} ± {s:.4f}" for m, s in
                     zip(summary["F1_Mean"], summary["F1_Std"].fillna(0.0))]
    summary[tcol] = [f"{m:.4f} ± {s:.4f}" for m, s in
                     zip(summary["Time_Mean"], summary["Time_Std"].fillna(0.0))]
    return summary.sort_values(["Dataset", "Accuracy_Mean"], ascending=[True, False])


def safe_append_to_excel_phase4(output_file, rows, lock):
    with lock:
        try:
            new_df = pd.DataFrame(rows)
            if os.path.exists(output_file):
                existing = pd.read_excel(output_file, sheet_name="Fold_Scores")
                folds_df = pd.concat([existing, new_df], ignore_index=True)
            else:
                folds_df = new_df
            summary_df = summarise_nested_folds(folds_df)
            temp_file = output_file.replace('.xlsx', '_temp.xlsx')
            with pd.ExcelWriter(temp_file, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                folds_df.to_excel(writer, sheet_name="Fold_Scores", index=False)
            if os.path.exists(output_file):
                os.remove(output_file)
            shutil.move(temp_file, output_file)
            return True
        except Exception as e:
            print(f"✗ Error writing Phase 4 to Excel: {e}")
            return False


def get_completed_tasks_phase4(output_file):
    """(dataset, method) pairs already scored. A pair is written atomically, so
    anything present is complete."""
    completed = set()
    if not output_file or not os.path.exists(output_file):
        return completed
    try:
        df = pd.read_excel(output_file, sheet_name="Fold_Scores")
        for dataset, method in zip(df["Dataset"], df["Method"]):
            completed.add((str(dataset), str(method)))
    except Exception as e:
        print(f"⚠ Could not read existing Phase 4 file: {e}")
    return completed


def phase4_nested_cv(base_folder=None):
    """Phase 4: repeated stratified nested cross-validation."""
    base_folder = base_folder or BASE_FOLDER
    # Noise is re-injected per fold from the clean data, so read the clean pool.
    if ACTIVE_NOISE_RATE:
        base_folder = CLEAN_BASE_FOLDER

    specs = nested_method_specs()
    print("\n" + "="*80)
    print(f"PHASE 4: NESTED CV ({NESTED_REPEATS} × {NESTED_OUTER_FOLDS}-fold outer, "
          f"{NESTED_INNER_FOLDS}-fold inner)")
    print("="*80 + "\n")
    print(f"Datasets pooled from : {base_folder}/  (data1 + data_te1 concatenated)")
    print(f"Methods              : {len(specs)}")
    print(f"Rare-class policy    : {NESTED_RARE_CLASS_POLICY}")
    if ACTIVE_NOISE_RATE:
        print(f"Label noise          : {ACTIVE_NOISE_RATE*100:g}% of each fold's TRAINING labels")
    print()

    existing_file = get_latest_file("results_phase4_nested_cv_")
    if existing_file:
        output_file = existing_file
        print(f"📂 Resuming Phase 4 from: {output_file}\n")
    else:
        timestamp = datetime.now().strftime("%y_%m_%d_%H%M%S")
        output_file = results_path(f"results_phase4_nested_cv_{timestamp}.xlsx")
        print(f"📝 Creating Phase 4 file: {output_file}\n")

    completed_tasks = get_completed_tasks_phase4(output_file)
    print(f"Already completed: {len(completed_tasks)} (dataset, method) pairs\n")

    subfolders = list_datasets(base_folder)
    manager = _MP_CTX.Manager()
    lock = manager.Lock()
    pending = []
    for dataset_name in subfolders:
        dataset_path = os.path.join(base_folder, dataset_name)
        for spec in specs:
            if (dataset_name, spec[0]) not in completed_tasks:
                pending.append((dataset_path, dataset_name, spec, output_file, lock,
                                ACTIVE_NOISE_RATE))

    total_tasks = len(subfolders) * len(specs)
    print(f"Total tasks: {total_tasks}\nRemaining: {len(pending)}\n")
    if not pending:
        print("✓ Phase 4 scores already complete — refreshing the tests.\n")
        append_nested_cv_tests(output_file)
        return output_file

    print("NOTE: k is re-tuned inside every outer fold — "
          f"expect roughly {NESTED_OUTER_FOLDS * NESTED_REPEATS}× the cost of one Phase 2+3 pass.\n")

    completed_count = len(completed_tasks)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=_MP_CTX) as executor:
        futures = {executor.submit(nested_cv_task, *task): task for task in pending}
        for future in as_completed(futures):
            if future.result():
                completed_count += 1
            print(f"Progress: {completed_count}/{total_tasks} "
                  f"({100*completed_count/max(total_tasks,1):.1f}%)")

    append_nested_cv_tests(output_file)
    print(f"\n✓ Phase 4 complete! Results: {output_file}\n")
    return output_file


# ============================================================================
# STATISTICAL TESTS ON THE NESTED-CV SCORES  (closing step of Phase 4)
#
#   Two levels, and they answer different questions:
#
#   1. Friedman + Nemenyi ACROSS datasets (rows = datasets, columns = methods),
#      the Demšar (2006) protocol — unchanged from Phase 11, except each cell
#      is now a mean over outer folds instead of a single split.  This is the
#      headline test; it stays at dataset level on purpose.
#
#   2. Pairwise comparison WITHIN each dataset, over the paired fold scores.
#      A plain paired t-test is invalid here: resampled folds share training
#      data, the variance is underestimated and p-values come out far too
#      small.  Nadeau & Bengio's corrected resampled t-test inflates the
#      variance by (1/n + n_test/n_train) to compensate; Wilcoxon is reported
#      next to it as the distribution-free check.
# ============================================================================

def corrected_resampled_ttest(diffs, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for overlapping CV folds."""
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return np.nan, np.nan
    variance = d.var(ddof=1)
    if variance == 0:
        return 0.0, 1.0
    t_stat = d.mean() / math.sqrt(variance * (1.0 / n + n_test / max(n_train, 1)))
    p_value = 2.0 * stats.t.sf(abs(t_stat), n - 1)
    return float(t_stat), float(p_value)


def model_of(method):
    """The neighbourhood rule a method name belongs to."""
    for name in ALL_MODELS:
        if method.startswith(name):
            return name
    return method.split("_")[0]


def nested_pairwise_tests(folds_df):
    """
    Per-dataset pairwise comparison, WITHIN one neighbourhood rule.

    Only methods built on the same rule are compared — CustomKNN against
    CustomKNN, MutualKNN against MutualKNN, SymmetricKNN against SymmetricKNN.
    Pitting CustomKNN against MutualKNN answers "which neighbourhood definition
    is better", which is not the claim under test here: the question is whether
    a given selection strategy beats another one *holding the rule fixed*
    (window-based vs adaptive vs a single tuned k).  The across-rule comparison
    is what the Friedman test over datasets is for.
    """
    records = []
    for dataset, sub in folds_df.groupby("Dataset"):
        pivot = sub.pivot_table(index=["Repeat", "Fold"], columns="Method", values="Accuracy")
        pivot = pivot.dropna(axis=0, how="any")
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            continue
        n_train = float(sub["N_Train"].mean())
        n_test = float(sub["N_Test"].mean())
        same_rule = [(a, b) for a, b in combinations(sorted(pivot.columns), 2)
                     if model_of(a) == model_of(b)]
        for a, b in same_rule:
            diffs = pivot[a].values - pivot[b].values
            t_stat, t_p = corrected_resampled_ttest(diffs, n_train, n_test)
            w_stat, w_p = wilcoxon_safe(pivot[a].values, pivot[b].values)
            records.append({
                "Dataset": dataset, "Model": model_of(a),
                "Method_A": a, "Method_B": b, "Folds": pivot.shape[0],
                "Mean_A": pivot[a].mean(), "Mean_B": pivot[b].mean(),
                "Mean_Difference": float(np.mean(diffs)),
                "Corrected_t": t_stat, "Corrected_t_pvalue": t_p,
                "Corrected_t_Significant": "Yes" if (t_p == t_p and t_p < 0.05) else "No",
                "Wilcoxon_stat": w_stat, "Wilcoxon_pvalue": w_p,
                "Wilcoxon_Significant": "Yes" if w_p < 0.05 else "No",
                "Winner": a if np.mean(diffs) > 0 else (b if np.mean(diffs) < 0 else "tie"),
            })
    return pd.DataFrame(records)


def append_nested_cv_tests(source):
    """
    Friedman / Nemenyi / corrected resampled t-tests, written into the workbook
    the nested CV just produced.

    Kept in the same file as the scores they describe: the tests are only ever
    meaningful for one particular set of folds, and splitting them across two
    workbooks made it possible to pair the wrong ones.
    """
    if not source or not os.path.exists(source):
        print("✗ No nested-CV results to test.\n")
        return None
    print("\n" + "-"*80)
    print("STATISTICAL TESTS ON THE NESTED-CV SCORES")
    print("-"*80 + "\n")
    folds_df = pd.read_excel(source, sheet_name="Fold_Scores")
    summary_df = summarise_nested_folds(folds_df)

    print("Step 1: mean ± std per dataset and method")
    print(summary_df[["Dataset", "Method", "Folds", "Accuracy", "F1"]].to_string(index=False))

    # ---- Friedman across datasets, on the fold-averaged accuracies ----
    matrix = summary_df.pivot(index="Dataset", columns="Method", values="Accuracy_Mean").dropna(axis=1)

    friedman_summary, statistic, p_value = None, np.nan, np.nan
    nemenyi_results, sig_pairs = None, None
    if matrix.shape[0] < 3 or matrix.shape[1] < 3:
        print(f"\n⚠ Friedman needs at least 3 datasets and 3 methods "
              f"(found {matrix.shape[0]} × {matrix.shape[1]}) — skipping the across-dataset test.")
        print("  The per-dataset pairwise tests below are still valid.")
    else:
        print("\nStep 2: Friedman test across datasets (fold-averaged accuracies)")
        friedman_summary, statistic, p_value = perform_friedman_test(matrix)
        if p_value < 0.05:
            print("\nStep 3: post-hoc Nemenyi")
            nemenyi_results, sig_pairs = perform_posthoc_nemenyi(matrix, friedman_summary)
        else:
            print("\nStep 3: skipping Nemenyi (Friedman not significant)")

    # ---- Per-dataset pairwise, over the paired folds ----
    print("\nStep 4: per-dataset pairwise tests, within each neighbourhood rule")
    pairwise = nested_pairwise_tests(folds_df)
    if pairwise.empty:
        per_rule = Counter(model_of(m) for m in folds_df["Method"].unique())
        if per_rule and max(per_rule.values()) < 2:
            print("  ⚠ Nothing to compare: each rule has one method "
                  + ", ".join(f"{k}={v}" for k, v in sorted(per_rule.items())) + ".")
            print("    These tests compare selection strategies while holding the rule")
            print("    fixed, so they need at least two methods per rule. Add families")
            print("    to phase4.methods in config.yaml (adaptive / sqrt / lastk).")
        else:
            print("  ⚠ Not enough overlapping folds for pairwise tests.")
    else:
        n_sig = (pairwise["Corrected_t_Significant"] == "Yes").sum()
        print(f"  {len(pairwise)} within-rule pairs tested, {n_sig} significant at "
              f"α=0.05 (corrected resampled t-test)")
        for rule, grp in pairwise.groupby("Model"):
            print(f"    {rule:14s} {len(grp):3d} pairs, "
                  f"{(grp['Corrected_t_Significant'] == 'Yes').sum()} significant")

    protocol = pd.DataFrame({
        "Setting": ["Outer CV", "Repeats", "Inner CV", "Seed", "Rare-class policy",
                    "Data pool", "Tie-breaking", "Across-dataset test",
                    "Within-dataset test", "Source file"],
        "Value": [f"Stratified {NESTED_OUTER_FOLDS}-fold", NESTED_REPEATS,
                  f"Stratified {NESTED_INNER_FOLDS}-fold (on outer-train only)",
                  NESTED_SEED, NESTED_RARE_CLASS_POLICY,
                  "data1 + data_te1 concatenated", CV_TIE_BREAKING,
                  "Friedman + Nemenyi post-hoc",
                  "Nadeau & Bengio corrected resampled t-test + Wilcoxon, "
                  "within each neighbourhood rule only",
                  os.path.basename(source)],
    })

    # Rewrite the workbook: the fold scores stay, the test sheets are refreshed.
    folds_raw = pd.read_excel(source, sheet_name="Fold_Scores")
    temp_file = source.replace('.xlsx', '_temp.xlsx')
    with pd.ExcelWriter(temp_file, engine='openpyxl') as writer:
        protocol.to_excel(writer, sheet_name="Protocol", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        matrix.to_excel(writer, sheet_name="Accuracy_Matrix")
        if friedman_summary is not None:
            friedman_summary.to_excel(writer, sheet_name="Rankings", index=False)
            pd.DataFrame({
                'Statistic': ['Friedman χ²', 'P-value', 'Degrees of Freedom', 'Significant (α=0.05)'],
                'Value': [statistic, p_value, matrix.shape[1] - 1,
                          'Yes' if p_value < 0.05 else 'No']
            }).to_excel(writer, sheet_name="Test_Statistics", index=False)
        if nemenyi_results is not None:
            nemenyi_results.to_excel(writer, sheet_name="Nemenyi_pvalues")
            if sig_pairs is not None:
                sig_pairs.to_excel(writer, sheet_name="Significant_Pairs", index=False)
        if not pairwise.empty:
            pairwise.to_excel(writer, sheet_name="Pairwise_Per_Dataset", index=False)
        folds_raw.to_excel(writer, sheet_name="Fold_Scores", index=False)
    os.remove(source)
    shutil.move(temp_file, source)

    print(f"\n✓ Tests written into: {source}")
    return source


def run_all_phases():
    """Run phases 1-13 on the currently active variant. Returns a summary dict."""
    def _enabled(phase):
        if not ENABLED_PHASES.get(phase, True):
            print(f"⊘ {phase} disabled in config.yaml — skipping.")
            return False
        return True

    print(f"\nPhase 1:  Initial k={INITIAL_K_VALUES}")
    print(f"Phase 2:  CV k range={min(CV_K_RANGE)}-{max(CV_K_RANGE)}, folds={CV_FOLDS}")
    print(f"Phase 3:  Test with best k")
    print(f"Phase 4:  Nested CV ({NESTED_REPEATS} × {NESTED_OUTER_FOLDS}-fold outer, "
          f"{NESTED_INNER_FOLDS}-fold inner) + tests"
          + ("" if ENABLED_PHASES.get("phase4") else "   [off by default]"))
    print(f"Phase 5:  Adaptive datasets k_max={ADAPTIVE_K_MAX_VALUES}")
    print(f"Phase 6:  Test adaptive datasets")
    print(f"Phase 7:  Adaptive datasets k_max=√(n/2)")
    print(f"Phase 8:  Test sqrt datasets")
    print(f"Phase 9:  Last stable k (k_max={LAST_K_MAX}, windows={STABILITY_WINDOWS})")
    print(f"Phase 10: Test last stable k datasets")
    print(f"Phase 11: Friedman statistical test")
    print(f"Phase 12: Wilcoxon pairwise tests")
    print(f"Phase 13: Combined CPU time report")
    disabled = [p for p, on in ENABLED_PHASES.items() if not on]
    if disabled:
        print(f"\n⊘ Disabled phases: {', '.join(disabled)}")
    print(f"\nWorkers: Phases 1-4={MAX_WORKERS}, Phases 5,7,9={BUILDER_WORKERS}, Phases 6,8,10={TESTER_WORKERS}")
    print(f"MP mode: {_MP_CTX.get_start_method()}")
    print(f"Backend: {get_backend().upper()}", end="")
    if get_backend() == "gpu":
        try:
            import torch
            dev = "CUDA" if torch.cuda.is_available() else ("MPS" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "CPU-fallback")
            print(f" (device: {dev})")
        except ImportError:
            print(" (torch not found — will fail at runtime)")
    else:
        print()
    print(f"Models:  {', '.join(MODELS)}")
    print(f"Timing:  {Timer.label()}", end="")
    if _TIMING_MODE == "gpu":
        try:
            import torch
            if torch.cuda.is_available():
                print(" (torch.cuda.Event)")
            else:
                print(" (time.perf_counter — no CUDA)")
        except ImportError:
            print(" (time.perf_counter — no torch)")
    else:
        print(" (time.process_time)")
    print("="*80)

    # ---- Phase 1 ----
    phase1_file = None
    if _enabled("phase1"):
        phase1_file = phase1_initial_experiments(BASE_FOLDER, INITIAL_K_VALUES)

    # ---- Phase 2 ----
    best_k_dict = {}
    if _enabled("phase2"):
        best_k_dict = phase2_find_best_k(BASE_FOLDER, CV_K_RANGE)

    # ---- Phase 3 ----
    phase3_file = None
    if _enabled("phase3"):
        if not best_k_dict:
            # Phase 2 may have been skipped; recover its choice from the grid on disk.
            best_k_dict = best_k_from_grid(get_latest_file("results_phase2_best_k_"))
        phase3_file = phase3_test_best_k(BASE_FOLDER, best_k_dict)

    # ---- Phase 4 (nested CV + its statistical tests) ----
    # Off by default: it re-tunes k inside every outer fold.
    phase4_file = phase4_nested_cv() if _enabled("phase4") else None

    # ---- Phase 5 ----
    if _enabled("phase5"):
        phase5_create_adaptive_datasets()

    # ---- Phase 6 ----
    phase6_file = None
    if _enabled("phase6"):
        phase6_file = phase6_test_adaptive_datasets()

    # ---- Phase 7 ----
    if _enabled("phase7"):
        phase7_create_sqrt_datasets()

    # ---- Phase 8 ----
    phase8_file = None
    if _enabled("phase8"):
        phase8_file = phase8_test_sqrt_datasets()

    # ---- Phase 9 ----
    if _enabled("phase9"):
        phase9_create_last_k_datasets()

    # ---- Phase 10 ----
    phase10_file = None
    if _enabled("phase10"):
        phase10_file = phase10_test_last_k_datasets()

    # ---- Phase 11 ----
    phase11_file = None
    if _enabled("phase11"):
        phase11_file = phase11_friedman_test()

    # ---- Phase 12 ----
    phase12_file = None
    if _enabled("phase12"):
        phase12_file = phase12_wilcoxon_pairwise_tests()

    # ---- Phase 13 ----
    # Always uses whatever result files exist on disk — safe to run even if
    # some earlier phases were disabled, will simply omit those sheets.
    phase13_file = None
    if _enabled("phase13"):
        phase13_file = phase13_combined_cpu_report()

    # ---- Per-dataset workbooks ----
    # Always built from whatever result files exist, so a partial run still
    # produces them. Set per_dataset_results: false in config.yaml to skip.
    per_dataset_folder = None
    if PER_DATASET_RESULTS:
        per_dataset_folder = export_per_dataset_results()
    else:
        print("⊘ per_dataset_results disabled in config.yaml — skipping.")

    return {
        "Phase 1  (k=%s)" % (INITIAL_K_VALUES,): phase1_file,
        "Phase 2  (Best k CV)": get_latest_file('results_phase2_best_k_'),
        "Phase 3  (Best k test)": phase3_file,
        "Phase 4  (Nested CV)": phase4_file,
        "Phase 5  (Adaptive create)": (', '.join(adaptive_folder(k) + '/'
                                                 for k in ADAPTIVE_K_MAX_VALUES)
                                       if ENABLED_PHASES.get('phase5') else None),
        "Phase 6  (Adaptive test)": phase6_file,
        "Phase 7  (Sqrt create)": (sqrt_folder() + '/'
                                   if ENABLED_PHASES.get('phase7') else None),
        "Phase 8  (Sqrt test)": phase8_file,
        "Phase 9  (Last_k create)": (last_k_folder() + '/'
                                     if ENABLED_PHASES.get('phase9') else None),
        "Phase 10 (Last_k test)": phase10_file,
        "Phase 11 (Friedman test)": phase11_file,
        "Phase 12 (Wilcoxon pairs)": phase12_file,
        "Phase 13 (CPU combined)": phase13_file,
        "Per-dataset workbooks": per_dataset_folder + '/' if per_dataset_folder else None,
        "All result files in": RESULTS_FOLDER + '/',
    }


def main():
    variants = build_variants()

    print("\n" + "="*80)
    print("KNN COMPLETE PIPELINE - ALL 12 PHASES")
    print("="*80)
    print(f"Workspace:         {WORKSPACE_DIR}/   (everything generated goes here)")
    print(f"Clean base folder: {CLEAN_BASE_FOLDER}   (outside the workspace, read-only)")
    print(f"Results folder:    {OUTPUT_FOLDER}/")
    if NOISE_ENABLED:
        print(f"Label noise:       {', '.join(f'{r*100:g}%' for r in NOISE_RATES)}"
              f" of training labels, seed {NOISE_SEED}")
        for rate in NOISE_RATES:
            print(f"Noisy datasets:    {noisy_base_folder(rate)}/")
    else:
        print("Label noise:       off")
    print(f"Runs: {len(variants)} — {', '.join(v.label for v in variants)}")
    migrate_legacy_results()

    # Build every noisy copy up front, before any phase touches them.
    if NOISE_ENABLED:
        create_all_noisy_datasets()

    summaries = []
    for variant in variants:
        activate_variant(variant)
        print("\n" + "#"*80)
        print(f"# RUN: {variant.label.upper()}")
        print(f"#   datasets: {variant.base_folder}/")
        print(f"#   results:  {variant.results_folder}/")
        print("#"*80)
        summaries.append((variant, run_all_phases()))

    print("\n" + "="*80)
    print("PIPELINE RUN COMPLETE!")
    print("="*80)
    for variant, summary in summaries:
        print(f"\n📊 Results Summary — {variant.label}:\n")
        for label, value in summary.items():
            print(f"  {label:28s} {value or 'skipped'}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # --export-per-dataset rebuilds the per-dataset workbooks from the result
    # files already on disk, without running any phase.
    # --make-noisy only builds the noisy copy of the datasets.
    if "--export-per-dataset" in sys.argv:
        for _variant in build_variants():
            activate_variant(_variant)
            migrate_legacy_results()
            print(f"\n### {_variant.label} → {RESULTS_FOLDER}/")
            export_per_dataset_results()
    elif "--nested-cv" in sys.argv:
        # Repeated stratified nested CV + its statistical tests, on their own.
        # Runs regardless of the phase4 toggle, and touches no other phase.
        if NOISE_ENABLED:
            create_all_noisy_datasets()
        for _variant in build_variants():
            activate_variant(_variant)
            migrate_legacy_results()
            print("\n" + "#"*80)
            print(f"# NESTED CV — {_variant.label.upper()}")
            print(f"#   results:  {_variant.results_folder}/")
            print("#"*80)
            phase4_nested_cv()
    elif "--make-noisy" in sys.argv:
        create_all_noisy_datasets()
    else:
        main()
