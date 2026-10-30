#!/usr/bin/env bash
# Source on kindred before any run:  source scripts/kindred_env.sh
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME/envs/magic"

# ~/.bashrc on kindred exports GPU=2 for other projects; MAGIC runs on GPU 0 unless MAGIC_GPU says otherwise.
export GPU="${MAGIC_GPU:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export SHARED="${SHARED:-$HOME/shared}"
export HF_HOME="${HF_HOME:-$SHARED/hf_cache}"
export MVTEC="$SHARED/datasets/mvtec_ad"
export SD2="$SHARED/models/sd2_inpainting"
export AUX="$SHARED/datasets/mvtec_ad_aux"   # generated object/anomaly masks, shared across projects
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
