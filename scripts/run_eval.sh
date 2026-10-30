#!/usr/bin/env bash
# Evaluation of the Phase 4 outputs, each step wrapped by perf_run.py.
#   tmux new -d -s magic_eval 'bash scripts/run_eval.sh'
cd "$(dirname "$0")/.."
source scripts/kindred_env.sh
mkdir -p runs/logs

GEN_SUFFIX=noise_1.0_guidance_scale_4.0_7.5_align

for ckpt in magic_lora_r16_noise_1.0 magic_full_noise_1.0; do
    gen="outputs/$ckpt/$GEN_SUFFIX"
    [ -d "$gen" ] || { echo "[eval] missing $gen"; continue; }
    tag=${ckpt%%_noise*}
    echo "[eval] $(date '+%F %T') KID $ckpt"
    python tools/perf_run.py --tag "$tag" -- python evaluation/cal_kid.py --real_path "$MVTEC" --generated_path "$gen" \
        > "runs/logs/eval_kid_$tag.log" 2>&1 || echo "[eval] KID $ckpt FAILED"
    echo "[eval] $(date '+%F %T') IC-LPIPS $ckpt"
    python tools/perf_run.py --tag "$tag" -- python evaluation/cal_ic_lpips.py --mvtec_path "$MVTEC" --gen_path "$gen" \
        > "runs/logs/eval_iclpips_$tag.log" 2>&1 || echo "[eval] IC-LPIPS $ckpt FAILED"
done

# Defect-type classification needs >= 2 defect classes per category: only screw (scratch_head vs scratch_neck).
gen="outputs/magic_lora_r16_noise_1.0/$GEN_SUFFIX"
for step in train test; do
    echo "[eval] $(date '+%F %T') classification $step screw"
    python tools/perf_run.py --tag magic_lora_r16_screw -- python "evaluation/$step-classification.py" \
        --mvtec_path "$MVTEC" --generated_data_path "$gen" --sample_name screw \
        --checkpoint_path runs/classification/magic_lora_r16 \
        > "runs/logs/eval_cls_${step}_lora.log" 2>&1 || echo "[eval] classification $step FAILED"
done
echo "[eval] $(date '+%F %T') done"
