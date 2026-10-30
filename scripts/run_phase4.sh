#!/usr/bin/env bash
# Phase 4 on kindred: LoRA on the representative targets, then a full fine-tune baseline,
# each train/inference wrapped by perf_run.py. A failed step is logged and the queue continues.
#   tmux new -d -s magic 'bash scripts/run_phase4.sh'
cd "$(dirname "$0")/.."
source scripts/kindred_env.sh
mkdir -p runs/logs

TARGETS="screw/scratch_head screw/scratch_neck metal_nut/scratch transistor/damaged_case cable/poke_insulation"
STEPS="${STEPS:-5000}"

aux_dirs() {  # screw masks ship with the repo; the other categories come from tools/make_*_masks.py
    if [ "$1" = screw ]; then
        MASK_DIR=dataset/anomaly_mask; NORMAL_MASKS=dataset/obj_foreground_mask
    else
        MASK_DIR="$AUX/anomaly_mask"; NORMAL_MASKS="$AUX/obj_foreground_mask"
    fi
}

train() {  # mode category defect
    local mode=$1 cat=$2 defect=$3 extra
    if [ "$mode" = lora ]; then extra="--use_lora --lora_rank 16 --mixed_precision fp16"; else extra="--mixed_precision no"; fi
    python tools/perf_run.py --tag "${mode}_${cat}_${defect}" -- \
        python run_train.py --base_dir "$MVTEC" --category "$cat" --defect_class "$defect" \
        --output_name magic --max_train_steps "$STEPS" --seed 0 $extra \
        > "runs/logs/train_${mode}_${cat}_${defect}.log" 2>&1 || echo "[phase4] train $mode $cat/$defect FAILED"
}

infer() {  # ckpt_name category defect tag
    local ckpt=$1 cat=$2 defect=$3 tag=$4
    aux_dirs "$cat"
    python tools/perf_run.py --tag "${tag}_${cat}_${defect}" -- \
        python inference.py --model_ckpt_root "model/$ckpt" --ddim_scheduler_root ./scheduler \
        --categories "$cat" --defect_classes "$defect" --dataset_type mvtec --text_noise_scale 1.0 \
        --defect_json dataset/CAMA_json_file/defect_classification.json \
        --match_json dataset/CAMA_json_file/matching_result.json \
        --normal_masks "$NORMAL_MASKS" --mask_dir "$MASK_DIR" \
        --gsi_use_schedule --gsi_schedule cosine --gsi_min 4.0 --gsi_max 7.5 --guidance_scale_outside 7.5 \
        --CAMA --base_dir "$MVTEC" --output_name "outputs/$ckpt" --seed 0 \
        > "runs/logs/infer_${tag}_${cat}_${defect}.log" 2>&1 || echo "[phase4] infer $tag $cat/$defect FAILED"
}

for t in $TARGETS; do
    echo "[phase4] $(date '+%F %T') LoRA train $t"; train lora "${t%/*}" "${t#*/}"
done
for t in $TARGETS; do
    echo "[phase4] $(date '+%F %T') LoRA inference $t"; infer magic_lora_r16_noise_1.0 "${t%/*}" "${t#*/}" lora
done
echo "[phase4] $(date '+%F %T') full fine-tune baseline screw/scratch_head"
train full screw scratch_head
infer magic_full_noise_1.0 screw scratch_head full
echo "[phase4] $(date '+%F %T') done"
