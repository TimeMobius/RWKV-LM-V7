#!/usr/bin/env bash
set -euo pipefail

# Run one short profile per context.  Each DATA_FILE must already have been
# generated with the matching --ctx-len and therefore contain ctx_len + 1
# tokens per document.  Failed contexts are recorded and the sweep continues.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTEXTS_STRING="${CONTEXTS:-86016 131072 163840 180224 188416 192512 194560 196608}"
PROFILE_STEPS="${PROFILE_STEPS:-2}"
GPU_PER_NODE="${GPU_PER_NODE:-8}"
STRATEGY="${STRATEGY:-deepspeed_stage_3_offload}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/Datasets/RWKV7_3B_SFT}"
DATA_PREFIX_TEMPLATE="${DATA_PREFIX_TEMPLATE:-${DATA_ROOT}/ctx{ctx}/SFT_RWKV7_3B}"
SWEEP_ROOT="${SWEEP_ROOT:-${REPO_DIR}/outs/3b-context-sweep}"

mkdir -p "$SWEEP_ROOT"
SUMMARY_FILE="$SWEEP_ROOT/summary.tsv"
printf 'ctx_len\tstatus\tprofile_dir\n' > "$SUMMARY_FILE"

for ctx in $CONTEXTS_STRING; do
  if (( ctx <= 0 || ctx % 16 != 0 )); then
    printf '%s\tinvalid\t-\n' "$ctx" | tee -a "$SUMMARY_FILE"
    continue
  fi
  data_file="${DATA_PREFIX_TEMPLATE//\{ctx\}/$ctx}"
  profile_dir="$SWEEP_ROOT/ctx${ctx}"
  echo "[context-sweep] profiling ctx_len=${ctx}, DATA_FILE=${data_file}"
  if CTX_LEN="$ctx" DATA_FILE="$data_file" PROJ_DIR="$profile_dir" \
      PROFILE_STEPS="$PROFILE_STEPS" GPU_PER_NODE="$GPU_PER_NODE" STRATEGY="$STRATEGY" \
      bash "$REPO_DIR/run_3b_sft_profile.sh"; then
    printf '%s\tpassed\t%s\n' "$ctx" "$profile_dir" | tee -a "$SUMMARY_FILE"
  else
    printf '%s\tfailed\t%s\n' "$ctx" "$profile_dir" | tee -a "$SUMMARY_FILE"
  fi
done

echo "[context-sweep] summary: $SUMMARY_FILE"
