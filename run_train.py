import os
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description="Training script for DreamBooth Inpainting.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory (e.g., ./mvtecad)")
    parser.add_argument("--defect_class", type=str, nargs="+", default=None, help="Specific defect class(es) (e.g., color)")
    parser.add_argument("--category", type=str, nargs="+", default=None)
    parser.add_argument("--soft_mask", default=None, action="store_true")
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="./model")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=os.environ.get("SD2", "stabilityai/stable-diffusion-2-inpainting"),
        help="SD2 inpainting weights (local dir or hub id). Default: $SD2",
    )
    parser.add_argument(
        "--text_noise_scale",
        type=float,
        default=1.0,
        help="Standard deviation of random noise added to encA/encB",
    )
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=None, help="Default: 5e-6 (full), 1e-4 (LoRA)")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    args = parser.parse_args()

    base_dir = args.base_dir
    categories = args.category if args.category is not None else os.listdir(base_dir)
    learning_rate = args.learning_rate or (1e-4 if args.use_lora else 5e-6)
    mode = f"lora_r{args.lora_rank}" if args.use_lora else "full"

    for category in categories:
        defect_classes = (
            args.defect_class
            if args.defect_class
            else [
                d
                for d in os.listdir(f"{base_dir}/{category}/ground_truth/")
                if os.path.isdir(os.path.join(f"{base_dir}/{category}/ground_truth/", d))
            ]
        )

        for defect in defect_classes:
            instance_data_dir = f"{base_dir}/{category}/test/{defect}"
            class_data_dir = f"{base_dir}/{category}/test/{defect}"
            mask_data_dir = f"{base_dir}/{category}/ground_truth/{defect}"
            output_dir = f"{args.output_root}/{args.output_name}_{mode}_noise_{args.text_noise_scale}/{category}/{defect}"

            command = (
                f"accelerate launch --num_processes 1 --mixed_precision {args.mixed_precision} train_dreambooth_noise.py "
                f'--pretrained_model_name_or_path="{args.pretrained_model}" '
                f'--instance_data_dir="{instance_data_dir}" '
                f'--class_data_dir="{class_data_dir}" '
                f'--output_dir="{output_dir}" '
                f"--prior_loss_weight=1.0 "
                f'--instance_prompt="a photo of sks defect" '
                f'--class_prompt="a photo of defect" '
                f"--resolution=512 "
                f"--train_batch_size={args.train_batch_size} "
                f"--gradient_accumulation_steps=1 "
                f"--learning_rate={learning_rate} "
                f'--lr_scheduler="constant" '
                f"--lr_warmup_steps=0 "
                f"--num_class_images=200 "
                f"--max_train_steps={args.max_train_steps} "
                f"--mixed_precision={args.mixed_precision} "
                f"--text_noise_scale={args.text_noise_scale} "
                f"--mvtecad "
                f'--mask_data_dir="{mask_data_dir}" '
                f"--center_crop "
            )
            if args.soft_mask:
                command += "--soft_mask "
            if args.seed is not None:
                command += f"--seed={args.seed} "
            if args.use_lora:
                command += f"--use_lora --lora_rank={args.lora_rank} "

            print(f"Running command for category '{category}', defect '{defect}'.", flush=True)
            subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    main()
