#!/usr/bin/env python
"""Build augmented anomaly masks for inference, mirroring the bundled dataset/anomaly_mask/screw layout.

Takes the same few-shot ground-truth masks the trainer uses (first len//3 after sorting by number)
and writes `--total` randomly rotated / scaled / flipped / shifted copies per defect as
<out_dir>/<category>/<defect>/<i>.jpg at --size x --size (bundled screw: 504 masks, 256 px).
Placement is random; CAMA relocates masks onto the object at inference time.

    python tools/make_aug_masks.py --mvtec $MVTEC --out_dir $AUX/anomaly_mask \
        --pairs metal_nut/scratch transistor/damaged_case cable/poke_insulation
"""
import argparse
import random
import re
from pathlib import Path

import cv2
import numpy as np


def last_number(path):
    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[-1]) if numbers else -1


def few_shot_masks(mvtec, category, defect):
    images = sorted((mvtec / category / "test" / defect).iterdir(), key=last_number)
    masks = sorted((mvtec / category / "ground_truth" / defect).iterdir(), key=last_number)
    return masks[: len(images) // 3]


def augment(mask, size, rng):
    """Cut the defect blob out, rotate/scale/flip it and paste it at a random position."""
    ys, xs = np.nonzero(mask)
    blob = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    if rng.random() < 0.5:
        blob = blob[:, ::-1]
    if rng.random() < 0.5:
        blob = blob[::-1, :]
    h, w = blob.shape
    diag = int(np.ceil(np.hypot(h, w))) + 2
    canvas = np.zeros((diag, diag), np.uint8)
    canvas[(diag - h) // 2:(diag - h) // 2 + h, (diag - w) // 2:(diag - w) // 2 + w] = blob
    matrix = cv2.getRotationMatrix2D((diag / 2, diag / 2), rng.uniform(0, 360), rng.uniform(0.8, 1.2))
    blob = cv2.warpAffine(canvas, matrix, (diag, diag), flags=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(blob)
    blob = blob[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]

    out = np.zeros_like(mask)
    h, w = blob.shape
    if h >= out.shape[0] or w >= out.shape[1]:
        blob = cv2.resize(blob, (min(w, out.shape[1] - 1), min(h, out.shape[0] - 1)), interpolation=cv2.INTER_NEAREST)
        h, w = blob.shape
    y, x = rng.randrange(0, out.shape[0] - h), rng.randrange(0, out.shape[1] - w)
    out[y:y + h, x:x + w] = blob
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_NEAREST)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mvtec", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--pairs", nargs="+", required=True, help="category/defect")
    parser.add_argument("--total", type=int, default=504)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    for pair in args.pairs:
        category, defect = pair.split("/")
        sources = [(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) * 255
                   for p in few_shot_masks(args.mvtec, category, defect)]
        sources = [m for m in sources if m.any()]
        out = args.out_dir / category / defect
        out.mkdir(parents=True, exist_ok=True)
        for i in range(args.total):
            mask = augment(sources[i % len(sources)], args.size, rng)
            cv2.imwrite(str(out / f"{i}.jpg"), mask)
        print(f"{pair}: {len(sources)} few-shot masks -> {args.total} masks in {out}")


if __name__ == "__main__":
    main()
