#!/usr/bin/env bash
#
# The centred longest-run rule on all fourteen datasets, on the GPU.
#
# Scores it against AdaNN (first correct k) and the fixed-window rule over the
# same stratified 10-fold design, so the three are paired on identical folds.
#
#   ./run_longest_stable.sh                  # everything, shuttle last
#   ./run_longest_stable.sh --skip-sh        # the other thirteen only
#   ./run_longest_stable.sh --only-sh        # shuttle on its own
#   ./run_longest_stable.sh --backend cpu
#   ./run_longest_stable.sh --windows "2 3 5"
#
# Shuttle is run last and into its own workspace. It is roughly as expensive as
# the other thirteen together — its symmetric derive alone is about nine minutes
# per fold at 52k training rows — so keeping it separate means a decision to
# stop early costs nothing that has already finished.
#
# Everything resumes: each (model, dataset) result is written when it completes
# and skipped on a re-run. Interrupting costs only the cell in flight.
#
set -euo pipefail
cd "$(dirname "$0")"

WORKSPACE="longest_stable_run"
BACKEND="gpu"
WINDOWS="2 3"
KMAX=30
SMALL="iris ecl bl pm ys bn ph txr ls tn pd mgt lir"
DO_SMALL=1; DO_SH=1
PYTHON="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --backend)   BACKEND="$2";   shift 2 ;;
    --windows)   WINDOWS="$2";   shift 2 ;;
    --k-max)     KMAX="$2";      shift 2 ;;
    --skip-sh)   DO_SH=0;        shift   ;;
    --only-sh)   DO_SMALL=0;     shift   ;;
    -h|--help)   sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# --- preflight --------------------------------------------------------------
fail=0
for f in longest_stable_k.py sensitivity_study.py knn_pipeline.py knn_algorithms.py; do
  [[ -f "$f" ]] || { echo "missing: $f" >&2; fail=1; }
done
[[ -d datasets ]] || { echo "missing: datasets/" >&2; fail=1; }
for d in $SMALL sh; do
  [[ -f "datasets/$d/data1" ]] || { echo "missing dataset: $d" >&2; fail=1; }
done
"$PYTHON" - <<'PY' || fail=1
import importlib.util, sys
missing = [m for m in ("numpy", "pandas", "matplotlib", "scipy", "sklearn", "yaml", "openpyxl")
           if importlib.util.find_spec(m) is None]
if missing:
    print("missing python packages: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
[[ "$fail" == "0" ]] || { echo; echo "preflight failed — nothing run." >&2; exit 1; }

if [[ "$BACKEND" == "gpu" ]]; then
  "$PYTHON" - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        print(f"gpu       : {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)")
    else:
        print("gpu       : torch present but CUDA unavailable — it will fall back to CPU")
except ImportError:
    print("gpu       : torch not installed — it will fall back to CPU")
PY
fi

mkdir -p "$WORKSPACE/logs"
STAMP=$(date '+%Y%m%d_%H%M%S')
echo "workspace : $WORKSPACE"
echo "backend   : $BACKEND    k_max = $KMAX    compared against windows: $WINDOWS"
echo "started   : $(date '+%Y-%m-%d %H:%M:%S')"
echo

started=$(date +%s)

if [[ "$DO_SMALL" == "1" ]]; then
  log="$WORKSPACE/logs/main_$STAMP.log"
  echo "── thirteen datasets → $log"
  stdbuf -oL -eL "$PYTHON" -W ignore longest_stable_k.py \
    --workspace "$WORKSPACE" --backend "$BACKEND" --k-max "$KMAX" \
    --windows $WINDOWS --datasets $SMALL 2>&1 | tee "$log" \
    | grep -E "^▶|^⊘|saved in|finished in|^  fig|^  table|vs |Traceback|Error" || true
fi

if [[ "$DO_SH" == "1" ]]; then
  log="$WORKSPACE/logs/sh_$STAMP.log"
  echo
  echo "── shuttle (separate workspace, the expensive one) → $log"
  stdbuf -oL -eL "$PYTHON" -W ignore longest_stable_k.py \
    --workspace "${WORKSPACE}_sh" --backend "$BACKEND" --k-max "$KMAX" \
    --windows $WINDOWS --datasets sh 2>&1 | tee "$log" \
    | grep -E "^▶|^⊘|saved in|finished in|^  fig|^  table|vs |Traceback|Error" || true
fi

echo
echo "=============================================================="
printf 'total elapsed: %d min\n' $(( ($(date +%s) - started) / 60 ))
for ws in "$WORKSPACE" "${WORKSPACE}_sh"; do
  [[ -d "$ws/results" ]] || continue
  echo
  echo "$ws/results/"
  ls -1 "$ws/results" 2>/dev/null | grep -v '^partials$' | sed 's/^/  /'
  n=$(ls -1 "$ws/results/partials" 2>/dev/null | wc -l)
  echo "  partials/ — $n (model, dataset) cells done"
done
echo
echo "Re-running skips whatever finished."
