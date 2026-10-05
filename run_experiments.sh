#!/usr/bin/env bash
#
# Full experiment set, for a server.
#
#   nested best-k          Phase 4, k retuned inside every outer fold
#   fixed k = 5            Phase 1
#   AdaNN k_max = 9, 30    Phases 5 (derive) + 6 (score)
#   tuned-w WAda           Phase 4, w retuned inside every outer fold
#
# for CustomKNN, MutualKNN and SymmetricKNN, on all 14 datasets, scored by
# stratified 10-fold cross-validation on the unscaled data with the min-max
# scaler fitted inside each training fold.
#
# Every phase draws its folds from the same seeded design, so any two methods
# can be compared as a paired test over identical folds.
#
# Resumable: each (dataset, method) pair is written atomically and skipped on a
# re-run, so interrupting and restarting costs only the pair in flight.
#
#   ./run_experiments.sh                    # everything, default workspace
#   ./run_experiments.sh --workspace srv1   # name the workspace
#   ./run_experiments.sh --datasets "iris ecl"   # a subset
#   ./run_experiments.sh --backend cpu      # see the note on backends below
#   ./run_experiments.sh --repeats 10       # 10x10-fold instead of 10-fold
#   ./run_experiments.sh --dry-run          # write the config, run nothing
#
set -euo pipefail
cd "$(dirname "$0")"

WORKSPACE="server_run"
DATASETS="iris ecl bl pm ys bn ph txr ls tn pd mgt lir sh"
MODELS='["CustomKNN", "MutualKNN", "SymmetricKNN"]'
BACKEND="gpu"
FOLDS=10
REPEATS=1
INNER=10
SEED=42
BASE="datasets"
DRY=0
PYTHON="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --datasets)  DATASETS="$2";  shift 2 ;;
    --backend)   BACKEND="$2";   shift 2 ;;
    --folds)     FOLDS="$2";     shift 2 ;;
    --repeats)   REPEATS="$2";   shift 2 ;;
    --inner)     INNER="$2";     shift 2 ;;
    --base)      BASE="$2";      shift 2 ;;
    --dry-run)   DRY=1;          shift   ;;
    -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

LOGDIR="$WORKSPACE/logs"
mkdir -p "$WORKSPACE" "$LOGDIR"

# --- config -----------------------------------------------------------------
# Timing follows the backend: process_time does not capture GPU kernels.
TIMING=$([[ "$BACKEND" == "gpu" ]] && echo gpu || echo cpu)
DS_YAML=$(printf '"%s", ' $DATASETS); DS_YAML="[${DS_YAML%, }]"

cat > "$WORKSPACE/config.yaml" <<CFGEOF
# Written by run_experiments.sh on $(date '+%Y-%m-%d %H:%M:%S')
base_folder: "$BASE"
output_folder: "results"
per_dataset_results: true

noise:
  enabled: false
  rates: [0.3]
  seed: 42
  run_clean: true

datasets:
  include: $DS_YAML
  exclude: []

models: $MODELS

backend: "$BACKEND"
timing: "$TIMING"
gpu_max_elements: 250000000

# One design for every phase: same folds, same scaler, so methods scored in
# different phases remain directly comparable.
cv:
  folds: $FOLDS
  repeats: $REPEATS
  bestk_folds: $FOLDS
  bestk_repeats: $REPEATS
  seed: $SEED
  scaling: "minmax"
  rare_class_policy: "warn"

phases:
  phase1:  true      # fixed k = 5
  phase2:  false
  phase3:  false
  phase4:  true      # nested best-k and tuned-w
  phase5:  true      # derive AdaNN neighbourhoods, k_max 9 and 30
  phase6:  true      # score them
  phase7:  false
  phase8:  false
  phase9:  false
  phase10: false
  phase11: true      # Friedman across datasets
  phase12: true      # Wilcoxon pairs
  phase13: true      # timing report

phase1:
  k_values: [5]

phase2:
  k_min: 1
  k_max: 30
  cv_folds: $INNER
  tie_breaking: "lowest_k"

phase4:
  outer_folds: $FOLDS
  inner_folds: $INNER
  repeats: $REPEATS
  seed: $SEED
  rare_class_policy: "warn"
  methods: ["bestk", "lastk_tuned"]
  w_values: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  fixed_k: [5]

phase5:
  k_max_values: [9, 30]

phase9:
  k_max: 30
  stability_windows: [3]

workers:
  main: -1
  builders: 1
  testers: 10
CFGEOF

echo "workspace : $WORKSPACE"
echo "datasets  : $DATASETS"
echo "protocol  : ${REPEATS}x${FOLDS}-fold outer, ${INNER}-fold inner, seed $SEED"
echo "backend   : $BACKEND"
echo "config    : $WORKSPACE/config.yaml"
echo
if [[ "$BACKEND" == "gpu" ]]; then
  cat <<'NOTE'
Note on backends. With backend=gpu the pipeline runs one worker per phase: each
spawned process would otherwise copy the whole CUDA runtime. The GPU speeds up
the distance matrices but not the row sorts or the per-class loops, so on small
and medium datasets many CPU workers usually finish sooner than one GPU worker.
The large ones (pd, mgt, lir, sh) are the other way round. Splitting the run is
often faster overall:

  ./run_experiments.sh --workspace srv_small --backend cpu \
      --datasets "iris ecl bl pm ys bn ph txr ls tn"
  ./run_experiments.sh --workspace srv_large --backend gpu \
      --datasets "pd mgt lir sh"

NOTE
fi

[[ "$DRY" == "1" ]] && { echo "dry run — nothing executed"; exit 0; }

STAMP=$(date '+%Y%m%d_%H%M%S')
LOG="$LOGDIR/run_$STAMP.log"
echo "log       : $LOG"
echo

# stdbuf keeps the log readable while the run is in flight; python buffers
# stdout hard when it is not a terminal.
stdbuf -oL -eL "$PYTHON" -W ignore knn_pipeline.py --workspace "$WORKSPACE" 2>&1 | tee "$LOG"

echo
echo "=============================================================="
echo "results : $WORKSPACE/results/"
ls -1 "$WORKSPACE/results/" 2>/dev/null | sed 's/^/  /'
echo
echo "Re-running skips whatever finished; only unfinished pairs are recomputed."
