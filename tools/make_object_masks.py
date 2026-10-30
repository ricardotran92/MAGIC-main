#!/usr/bin/env python
"""Foreground (object) masks of the normal training images with U-2-Net, for CAMA.

Uses the U-2-Net model through `rembg` (weights cached in $U2NET_HOME, default
~/shared/models/u2net so other projects reuse them). Writes
<out_dir>/<category>/train/good/<name>_mask.png, the layout of dataset/obj_foreground_mask.

    python tools/make_object_masks.py --mvtec $MVTEC --out_dir $AUX/obj_foreground_mask \
        --categories metal_nut transistor cable
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("U2NET_HOME", str(Path.home() / "shared" / "models" / "u2net"))

import numpy as np
from PIL import Image
from rembg import new_session, remove


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mvtec", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--model", default="u2net")
    args = parser.parse_args()

    session = new_session(args.model)
    for category in args.categories:
        images = sorted((args.mvtec / category / "train" / "good").glob("*.png"))
        out = args.out_dir / category / "train" / "good"
        out.mkdir(parents=True, exist_ok=True)
        coverage = []
        for path in images:
            mask = np.array(remove(Image.open(path).convert("RGB"), session=session, only_mask=True))
            mask = ((mask > 127) * 255).astype(np.uint8)
            coverage.append(mask.mean() / 255)
            Image.fromarray(mask).save(out / f"{path.stem}_mask.png")
        print(f"{category}: {len(images)} masks -> {out} (foreground {np.mean(coverage):.1%} mean, "
              f"{np.min(coverage):.1%} min)")


if __name__ == "__main__":
    main()
