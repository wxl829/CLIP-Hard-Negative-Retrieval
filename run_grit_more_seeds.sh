#!/usr/bin/env bash
# Run paired Random-vs-GRIT CLIP experiments for seeds 42, 123 and 2026.
# Execute from the project root. Writes fresh runs under outputs/reproduced/d7.
set -Eeuo pipefail

if [ ! -f outputs/d5/embeddings/metadata.json ]; then
  echo 'ERROR: run from the project root after extracting training embeddings.' >&2
  exit 1
fi

GPU_ID="${GPU_ID:-0}"
RUN_ROOT="${RUN_ROOT:-outputs/reproduced/d7}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-src/21_train_clip_grouped.py}"
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "ERROR: Training script not found: $TRAIN_SCRIPT" >&2
  exit 1
fi

if [ -n "${BUILD_SCRIPT:-}" ]; then
  [ -f "$BUILD_SCRIPT" ] || { echo "ERROR: Builder script not found: $BUILD_SCRIPT" >&2; exit 1; }
else
  mapfile -t candidates < <(find src -maxdepth 1 -type f -name '*build_grit_batches.py' | sort)
  if [ ${#candidates[@]} -ne 1 ]; then
    printf 'ERROR: Expected one src/*build_grit_batches.py, found %s. Set BUILD_SCRIPT=src/<actual_name>.\n' "${#candidates[@]}" >&2
    printf 'Found: %s\n' "${candidates[*]:-none}" >&2
    exit 1
  fi
  BUILD_SCRIPT="${candidates[0]}"
fi

mkdir -p "$RUN_ROOT/logs"

echo "Builder: $BUILD_SCRIPT"
echo "Trainer: $TRAIN_SCRIPT"
echo "Physical GPU: $GPU_ID"

for seed in 42 123 2026; do
  group_dir="$RUN_ROOT/grit_batches_seed${seed}"
  group_file="$group_dir/grouped_batches.json"
  if [ -f "$group_file" ]; then
    echo "Existing group file for seed $seed; verifying rather than overwriting."
  else
    if [ -d "$group_dir" ]; then
      echo "ERROR: Partial group directory exists: $group_dir. Inspect it first." >&2
      exit 1
    fi
    python "$BUILD_SCRIPT" \
      --batch-size 16 --epochs 10 --seed "$seed" --output-dir "$group_dir" \
      2>&1 | tee "$RUN_ROOT/logs/build_seed${seed}.log"
  fi

  python - "$group_file" "$seed" <<'PY'
import json, sys
from pathlib import Path
p, expected = Path(sys.argv[1]), int(sys.argv[2])
a = json.loads(p.read_text(encoding='utf-8'))
assert int(a['seed']) == expected, f'Seed mismatch in {p}'
assert int(a['num_images']) == 5000
assert int(a['batch_size']) == 16
assert len(a['epochs']) >= 10, f'Insufficient epochs in {p}'
print(f'Verified group file: seed={expected}, epochs={len(a["epochs"])}')
PY

  for sampler in random grit; do
    if [ "$seed" == 42 ]; then
      if [ "$sampler" == random ]; then run_name=ft_random_matched; else run_name=ft_grit_style; fi
    else
      run_name="ft_${sampler}_seed${seed}"
    fi
    output="$RUN_ROOT/$run_name"
    if [ -e "$output" ]; then
      echo "ERROR: Output already exists; refusing to overwrite: $output" >&2
      echo "If this is a finished run, move to the next command manually. If interrupted, inspect files before deciding whether to restart." >&2
      exit 1
    fi
    echo "=== START seed=$seed sampler=$sampler ==="
    CUDA_VISIBLE_DEVICES="$GPU_ID" python "$TRAIN_SCRIPT" \
      --group-file "$group_file" \
      --sampler "$sampler" \
      --epochs 10 \
      --batch-size 16 \
      --lr 1e-6 \
      --seed "$seed" \
      --output "$output" \
      2>&1 | tee "$RUN_ROOT/logs/train_${sampler}_seed${seed}.log"
  done
done

python summarize_grit_seeds.py --root "$RUN_ROOT"
