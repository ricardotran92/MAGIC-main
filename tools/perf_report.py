#!/usr/bin/env python
"""Summarise the perf_run.py reports into one markdown page (LoRA vs full fine-tune).

    python tools/perf_report.py                      # runs/perf/*.json -> runs/perf/REPORT.md
    python tools/perf_report.py --include-smoke --out -

Pure post-processing of the JSON reports: no GPU, safe to run anywhere.
"""
import argparse
import json
import re
from pathlib import Path


def load_reports(report_dir, include_smoke):
    reports = []
    for path in sorted(Path(report_dir).glob("*.json")):
        r = json.loads(path.read_text())
        if not include_smoke and r.get("tag", "").startswith("smoke"):
            continue
        r["_file"] = path.name
        reports.append(r)
    return reports


def fmt(value, digits=2):
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(c) if not isinstance(c, str) else c for c in row) + " |" for row in rows]
    return "\n".join(lines)


def mib_to_gib(mib):
    return None if mib is None else mib / 1024


def target_of(tag):
    """'lora_screw_scratch_head' -> ('lora', 'screw/scratch_head'); categories may contain '_' (metal_nut)."""
    mode, rest = tag.split("_", 1)
    for cat in ("metal_nut", "screw", "transistor", "cable", "hazelnut"):
        if rest.startswith(cat + "_"):
            return mode, f"{cat}/{rest[len(cat) + 1:]}"
    return mode, rest


def system_section(r):
    s, gpu = r["system"], r["before"]["selected_gpu"]
    pk = s["packages"]
    return "\n".join([
        "## Machine",
        f"- Host: `{s['hostname']}` ({s['os']}), Python {s['python']}",
        f"- CPU: {s['cpu_model']}, {s['cpu_physical']} cores / {s['cpu_logical']} threads; "
        f"RAM {s['ram_total_gib']} GiB",
        f"- GPU {gpu['index']}: {gpu['name']}, {gpu['memory.total'] / 1024:.0f} GiB, "
        f"driver {gpu['driver_version']}, power limit {gpu['power.limit']:.0f} W",
        f"- torch {pk.get('torch')}, diffusers {pk.get('diffusers')}, peft {pk.get('peft')}, "
        f"transformers {pk.get('transformers')}, accelerate {pk.get('accelerate')}",
        f"- Usage sampled every {r['during']['interval_s']:.0f} s; values below are mean / peak over the run.",
    ])


def train_section(reports):
    rows = []
    for r in reports:
        w, d = r["workload"], r["during"]
        mode, target = target_of(r["tag"])
        rows.append([target, w.get("mode") or mode, w.get("lora_rank") or "–", fmt((w.get("few_shot_examples") or [None])[0]),
                     w["done"], r["wall_hms"], fmt(w.get("seconds_per_unit"), 3), fmt(w.get("images_per_s")),
                     fmt(d["gpu_util"]["mean"], 0), fmt(mib_to_gib(d["proc_gpu_mem"]["max"]), 1),
                     fmt(d["gpu_power"]["mean"], 0), fmt(d["proc_rss"]["max"], 1), fmt(w.get("checkpoint_gib"), 3)])
    header = ["target", "mode", "rank", "few-shot imgs", "steps", "wall", "s/step", "img/s", "GPU util %",
              "VRAM peak GiB", "power W", "RSS peak GiB", "ckpt GiB"]
    return "## Training (bs 4, 5000 steps)\n\n" + table(header, rows)


def infer_section(reports):
    rows = []
    for r in reports:
        w, d = r["workload"], r["during"]
        mode, target = target_of(r["tag"])
        rows.append([target, mode, w["done"], r["wall_hms"], fmt(w.get("seconds_per_unit")),
                     fmt(w.get("throughput_per_s"), 3), fmt(d["gpu_util"]["mean"], 0),
                     fmt(mib_to_gib(d["proc_gpu_mem"]["max"]), 1), fmt(d["gpu_power"]["mean"], 0)])
    header = ["target", "mode", "images", "wall", "s/image", "image/s", "GPU util %", "VRAM peak GiB", "power W"]
    return "## Inference (DDIM 50 steps, CAMA, gsi cosine 4.0-7.5)\n\n" + table(header, rows)


def quality_section(kid, lpips, cls):
    """kid/lpips: {mode: {target: value}} from the latest eval report per mode."""
    targets = sorted({t for m in (kid, lpips) for v in m.values() for t in v})
    modes = sorted({m for d in (kid, lpips) for m in d}, key=lambda m: (m != "lora", m))
    header = ["target"] + [f"KID×1000 ({m}) ↓" for m in modes] + [f"IC-LPIPS ({m}) ↑" for m in modes]
    rows = [[t] + [fmt(kid.get(m, {}).get(t)) for m in modes] + [fmt(lpips.get(m, {}).get(t), 4) for m in modes]
            for t in targets]
    out = ["## Quality", "", table(header, rows)]
    if cls:
        out += ["", "Defect-type classification (ResNet34 trained on generated screw images, tested on the real "
                "MVTec test images not used as few-shot):", ""]
        out += [f"- {tag}: average accuracy {fmt(acc, 2)} %" for tag, acc in cls.items()]
    return "\n".join(out)


def eval_metrics(reports):
    kid, lpips, cls, rows = {}, {}, {}, []
    for r in reports:
        metrics = r["workload"].get("metrics") or {}
        mode = r["tag"].replace("magic_", "").split("_r16")[0]  # magic_lora_r16 -> lora, magic_full -> full
        if r["script"] == "cal_kid.py":
            kid[mode] = {k.split(" ", 1)[1]: v for k, v in metrics.items() if " " in k and "/" in k}
        elif r["script"] == "cal_ic_lpips.py":
            lpips[mode] = {k.split(" ")[-1]: v for k, v in metrics.items() if "/" in k}
        elif r["script"] == "test-classification.py":
            cls[r["tag"]] = metrics.get("avg_accuracy")
        rows.append([r["script"], r["tag"], r["workload"]["done"], r["wall_hms"],
                     fmt(r["workload"].get("throughput_per_s")), fmt(mib_to_gib(r["during"]["proc_gpu_mem"]["max"]), 1)])
    header = ["script", "tag", "work", "wall", "unit/s", "VRAM peak GiB"]
    return kid, lpips, cls, "## Evaluation runs\n\n" + table(header, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-dir", default="runs/perf")
    parser.add_argument("--out", default=None, help="Markdown path (default <report-dir>/REPORT.md, '-' for stdout)")
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()

    reports = [r for r in load_reports(args.report_dir, args.include_smoke) if r.get("status") == "completed"]
    if not reports:
        raise SystemExit(f"no completed reports in {args.report_dir}")
    train = [r for r in reports if r["script"] in ("run_train.py", "train_dreambooth_noise.py")]
    infer = [r for r in reports if r["script"] == "inference.py"]
    evals = [r for r in reports if r["script"].startswith(("cal_", "train-class", "test-class"))]

    sections = ["# MAGIC on kindred: LoRA vs full fine-tune", "",
                f"Generated by `tools/perf_report.py` from {len(reports)} reports in `{args.report_dir}` "
                f"(commits {', '.join(sorted({r['git']['commit'] for r in reports}))}).", "",
                system_section(reports[0])]
    if train:
        sections += ["", train_section(train)]
    if infer:
        sections += ["", infer_section(infer)]
    if evals:
        kid, lpips, cls, eval_table = eval_metrics(evals)
        sections += ["", quality_section(kid, lpips, cls), "", eval_table]
    text = "\n".join(sections) + "\n"

    if args.out == "-":
        print(text)
    else:
        out = Path(args.out or Path(args.report_dir) / "REPORT.md")
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
