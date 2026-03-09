#!/bin/bash
# Download LIBERO benchmark datasets (4 suites) from HuggingFace.
# Usage: bash script/download_libero.sh [DATA_ROOT]
#
# HF repo: yifengzhu-hf/LIBERO-datasets
# Note: LIBERO-Long is stored as "libero_10" on HF; we rename it to
#       "libero_long" locally for consistency with the training pipeline.
set -euo pipefail

DATA_ROOT="${1:-data/libero}"
REPO_ID="yifengzhu-hf/LIBERO-datasets"
mkdir -p "$DATA_ROOT"

# (local_name  hf_folder_name)
declare -A SUITE_MAP=(
    ["libero_spatial"]="libero_spatial"
    ["libero_object"]="libero_object"
    ["libero_goal"]="libero_goal"
    ["libero_long"]="libero_10"
)

echo "=== Downloading LIBERO datasets to $DATA_ROOT ==="
echo "    HF repo: $REPO_ID"

# Detect the HF download command
if command -v hf &> /dev/null; then
    HF_CMD="hf"
elif command -v huggingface-cli &> /dev/null; then
    HF_CMD="huggingface-cli"
else
    echo "Neither 'hf' nor 'huggingface-cli' found. Installing huggingface_hub[cli]..."
    pip install -q "huggingface_hub[cli]"
    HF_CMD="hf"
fi
echo "    Using CLI: $HF_CMD"

for local_name in libero_spatial libero_object libero_goal libero_long; do
    hf_name="${SUITE_MAP[$local_name]}"
    dest="$DATA_ROOT/$local_name"

    if [ -d "$dest" ] && [ "$(find "$dest" -name '*.hdf5' 2>/dev/null | head -1)" ]; then
        count=$(find "$dest" -name "*.hdf5" | wc -l)
        echo "[SKIP] $local_name already has $count HDF5 files at $dest"
        continue
    fi

    echo "[DOWNLOAD] $local_name (HF: $hf_name) -> $dest"
    mkdir -p "$dest"

    $HF_CMD download "$REPO_ID" \
        --repo-type dataset \
        --include "${hf_name}/*" \
        --local-dir "$DATA_ROOT"

    # If HF folder name differs from local name, move files
    if [ "$hf_name" != "$local_name" ] && [ -d "$DATA_ROOT/$hf_name" ]; then
        mv "$DATA_ROOT/$hf_name"/* "$dest/" 2>/dev/null || true
        rmdir "$DATA_ROOT/$hf_name" 2>/dev/null || true
        echo "  Renamed $hf_name -> $local_name"
    fi
done

echo ""
echo "=== Download complete. Directory structure ==="
for local_name in libero_spatial libero_object libero_goal libero_long; do
    count=$(find "$DATA_ROOT/$local_name" -name "*.hdf5" 2>/dev/null | wc -l)
    echo "  $local_name: $count HDF5 files"
done
echo ""
echo "Next step: run VAE pre-encoding with:"
echo "  python wan_va/data/preprocess_libero.py --data-root $DATA_ROOT --output-root data/libero_preprocessed"
