import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from skimage import color
from skimage.color import deltaE_cie76


def _linear_to_srgb(img_float):
    img = np.clip(img_float, 0.0, 1.0)
    return np.where(img <= 0.0031308,
                    12.92 * img,
                    1.055 * img ** (1.0 / 2.4) - 0.055)


def _psnr_channel(pred, gt):
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def psnr_rgb(pred_bgr, gt_bgr):
    return _psnr_channel(pred_bgr / 255.0, gt_bgr / 255.0)


def psnr_luminance(pred_bgr, gt_bgr):
    p_y = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64) / 255.0
    g_y = cv2.cvtColor(gt_bgr,  cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64) / 255.0
    return _psnr_channel(p_y, g_y)


def ssim_score(pred_bgr, gt_bgr):
    from skimage.metrics import structural_similarity as ssim
    p = _linear_to_srgb(pred_bgr.astype(np.float32) / 255.0)
    g = _linear_to_srgb(gt_bgr.astype(np.float32)   / 255.0)
    return float(ssim(p, g, channel_axis=2, data_range=1.0))


def delta_e76(pred_bgr, gt_bgr):
    pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gt_rgb   = cv2.cvtColor(gt_bgr,  cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    pred_rgb = _linear_to_srgb(pred_rgb)
    gt_rgb   = _linear_to_srgb(gt_rgb)
    return float(np.mean(deltaE_cie76(color.rgb2lab(pred_rgb), color.rgb2lab(gt_rgb))))


def list_runs(extra):
    runs_dir = os.path.join("./runs", extra)
    if not os.path.isdir(runs_dir):
        print(f"ERROR: '{runs_dir}' not found. Run from inside paper_code/.")
        sys.exit(1)
    return natsorted(
        [d for d in os.listdir(runs_dir)
         if os.path.isdir(os.path.join(runs_dir, d))]
    )


def pick_run_interactively(extra):
    runs = list_runs(extra)
    if not runs:
        print(f"No experiment folders found in runs/{extra}/.")
        sys.exit(1)
    print(f"\nAvailable experiments in runs/{extra}/:")
    print(f"  [all] — evaluate every run in this folder")
    for i, r in enumerate(runs):
        print(f"  [{i}] {r}")
    idx = input("\nEnter index, name, or 'all': ").strip()
    if idx == "all":
        return runs
    if idx.isdigit():
        return runs[int(idx)]
    if idx in runs:
        return idx
    print(f"Unknown experiment '{idx}'.")
    sys.exit(1)


def find_checkpoint(opt, mode):
    from utils.our_utils import find_best_val_weights, find_last_weights

    run_dir = opt["root_dir"]

    if mode == "best":
        candidate = os.path.join(run_dir, "best_psnr.pth")
        if os.path.isfile(candidate):
            return candidate, "best_psnr"
        print("  best_psnr.pth not found, falling back to best_val.")
        mode = "best_val"

    if mode == "best_val":
        path, it = find_best_val_weights(opt)
        return path, f"best_val_iter{it}"

    if mode == "last":
        path, it = find_last_weights(opt)
        return path, f"last_iter{it}"

    print(f"Unknown checkpoint mode '{mode}'.")
    sys.exit(1)


def run_inference_split(opt, model, phase, images_dir):
    from data import create_dataloader, create_dataset

    dataset_opt = opt["datasets"][phase]
    dataset_opt["phase"] = phase
    dataset_opt["scale"] = opt["scale"]
    dataset_opt["name"] = "TestSet"

    opt["path"]["test_images_dir"] = images_dir
    os.makedirs(images_dir, exist_ok=True)

    dataset = create_dataset(dataset_opt)
    loader  = create_dataloader(
        dataset, dataset_opt,
        num_gpu=opt["num_gpu"], dist=opt["dist"],
        sampler=None, seed=opt["manual_seed"]
    )
    print(f"  {phase.capitalize()} images: {len(dataset)}")

    phase_cfg = opt.get(phase, opt.get("val", {}))
    rgb2bgr   = phase_cfg.get("rgb2bgr", True)
    use_image = phase_cfg.get("use_image", True)
    model.validation(
        loader, current_iter=0, tb_logger=None,
        save_img=True, rgb2bgr=rgb2bgr, use_image=use_image
    )


def compute_all_metrics(images_dir):
    import pyiqa
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    lpips_metric = pyiqa.create_metric("lpips").to(device)

    pred_files = natsorted([
        f for f in os.listdir(images_dir)
        if f.endswith(".png")
        and "_gt"       not in f
        and "_in"       not in f
        and "_gamma"    not in f
        and "_original" not in f
        and "_wb"       not in f
    ])

    if not pred_files:
        print(f"  WARNING: no prediction images found in {images_dir}")
        return pd.DataFrame()

    records = []
    for fname in pred_files:
        pred_path = os.path.join(images_dir, fname)
        gt_path   = os.path.join(images_dir, fname[:-4] + "_gt.png")
        pred = cv2.imread(pred_path)
        gt   = cv2.imread(gt_path)

        if pred is None or gt is None:
            print(f"  SKIP (missing file): {fname}")
            continue

        try:
            illum = int(fname.split("B_")[-1].split(".")[0])
        except (ValueError, IndexError):
            illum = -1

        records.append({
            "image":    fname,
            "illum":    illum,
            "PSNR-L":   round(psnr_luminance(pred, gt),                   4),
            "PSNR-C":   round(psnr_rgb(pred, gt),                         4),
            "SSIM":     round(ssim_score(pred, gt),                        4),
            "DeltaE76": round(delta_e76(pred, gt),                         4),
            "LPIPS":    round(lpips_metric(pred_path, gt_path).item(),     4),
        })

    return pd.DataFrame(records)


def apply_gamma_to_saved_images(images_dir, gamma=2.2):
    png_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]
    for fname in png_files:
        path = os.path.join(images_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        img_g = np.clip(img.astype(np.float32) / 255.0, 0, 1) ** (1.0 / gamma)
        cv2.imwrite(path, (img_g * 255).round().astype(np.uint8))
    print(f"  Gamma {gamma} applied to {len(png_files)} images in {images_dir}")


def print_and_save_results(df, split_label, run_dir, dsname):
    if df.empty:
        print(f"  No results for {split_label}.")
        return

    metric_cols = ["PSNR-L", "PSNR-C", "SSIM", "DeltaE76", "LPIPS"]

    print(f"\nPer-image results ({split_label}):")
    print(df.to_string(index=False))

    known = df[df["illum"] >= 0]
    per_illum = None
    if known["illum"].nunique() > 1:
        per_illum = (
            known.groupby("illum")[metric_cols]
            .mean().sort_index().round(4)
        )
        print(f"\n{'─'*60}")
        print(f"Mean per illumination level — {split_label}:")
        print(per_illum.to_string())

    means = df[metric_cols].mean()
    print(f"\n{'─'*60}")
    print(f"Overall mean — {split_label}:")
    print(f"  PSNR-L   (↑) : {means['PSNR-L']:.4f}")
    print(f"  PSNR-C   (↑) : {means['PSNR-C']:.4f}")
    print(f"  SSIM     (↑) : {means['SSIM']:.4f}")
    print(f"  DeltaE76 (↓) : {means['DeltaE76']:.4f}")
    print(f"  LPIPS    (↓) : {means['LPIPS']:.4f}")
    print(f"{'─'*60}\n")

    tag     = split_label.lower()
    out_csv = os.path.join(run_dir, f"eval_metrics_{dsname}_{tag}.csv")
    df_out  = pd.concat([df, means.rename("MEAN").to_frame().T], ignore_index=True)
    df_out.to_csv(out_csv, index=False)
    print(f"  Per-image CSV : {out_csv}")

    if per_illum is not None:
        out_illum = os.path.join(run_dir, f"eval_metrics_{dsname}_{tag}_per_illum.csv")
        per_illum.to_csv(out_illum)
        print(f"  Per-illum CSV : {out_illum}")


def make_scene_visualizations(images_dir, display_gamma=1/2.2, thumb_w=512):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vis_dir = os.path.join(images_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    pred_files = natsorted([
        f for f in os.listdir(images_dir)
        if f.endswith(".png")
        and "_gt"       not in f
        and "_in"       not in f
        and "_gamma"    not in f
        and "_original" not in f
        and "_wb"       not in f
    ])

    scenes = {}
    for fname in pred_files:
        stem = fname[:-4]
        try:
            illum = int(stem.split("B_")[-1])
            scene = stem[: stem.rfind("-B_")]
        except (ValueError, IndexError):
            continue
        scenes.setdefault(scene, []).append((illum, stem))

    print(f"  Building scene grids for {len(scenes)} scenes → {vis_dir}")

    def load_thumb(path):
        img = cv2.imread(path)
        if img is None:
            return np.zeros((thumb_w * 2 // 3, thumb_w, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        new_h = int(round(h * thumb_w / w))
        img = cv2.resize(img, (thumb_w, new_h), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if display_gamma != 1.0:
            img = (np.clip(img.astype(np.float32) / 255.0, 0, 1) ** display_gamma * 255).astype(np.uint8)
        return img

    col_labels = ["Input", "Prediction", "GT"]

    for scene, items in scenes.items():
        items.sort(key=lambda x: x[0])
        n_rows, n_cols = len(items), 3

        sample_img = load_thumb(os.path.join(images_dir, items[0][1] + ".png"))
        cell_h_in  = sample_img.shape[0] / 100
        cell_w_in  = thumb_w / 100

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * cell_w_in, n_rows * cell_h_in),
                                 squeeze=False)

        for col_idx, label in enumerate(col_labels):
            axes[0][col_idx].set_title(label, fontsize=9, fontweight="bold", pad=4)

        for row_idx, (illum, stem) in enumerate(items):
            paths = [
                os.path.join(images_dir, stem + "_in.png"),
                os.path.join(images_dir, stem + ".png"),
                os.path.join(images_dir, stem + "_gt.png"),
            ]
            for col_idx, path in enumerate(paths):
                ax = axes[row_idx][col_idx]
                ax.imshow(load_thumb(path))
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            axes[row_idx][0].set_ylabel(
                f"B={illum}", fontsize=8, rotation=0, labelpad=36, va="center"
            )

        fig.suptitle(scene, fontsize=8, y=1.005)
        plt.tight_layout(pad=0.3)
        fig.savefig(os.path.join(vis_dir, f"{scene}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  Done — {len(scenes)} grids saved to {vis_dir}")


def evaluate_run(run_name, extra, args):
    from utils.our_utils import LoadParams
    from models import create_model

    print(f"\n{'='*60}")
    print(f"  Experiment : {run_name}")
    print(f"  Extra      : {extra}")

    opt = LoadParams(run_name, extra=extra + "/")
    opt["path"]["visualization"]    = opt["output_dir"]
    opt["path"]["experiments_root"] = opt["root_dir"]
    opt["is_train"] = False
    opt["dist"]     = False

    ckpt_path, ckpt_label = find_checkpoint(opt, mode=args.checkpoint)
    print(f"  Checkpoint : {ckpt_path}  ({ckpt_label})")
    opt["path"]["pretrain_network_g"] = ckpt_path

    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = True

    model = create_model(opt)
    print(f"{'='*60}\n")

    run_dir   = opt["root_dir"]
    gamma_tag = "%.2f_%.2f" % (
        opt["datasets"]["test"]["gamma_train"],
        opt["datasets"]["test"]["gamma_out"],
    )

    splits_to_run = ["test", "val"] if args.split == "both" else [args.split]
    run_means = {}

    for split in splits_to_run:
        images_dir = os.path.join(
            run_dir, f"Eval_images_{gamma_tag}_{args.dsname}_{split}"
        )
        print(f"{'─'*60}")
        print(f"  Split      : {split.upper()}")
        print(f"  Images dir : {images_dir}")

        if not args.skip_infer:
            print("  Running inference …")
            run_inference_split(opt, model, split, images_dir)
        else:
            print("  Skipping inference (--skip_infer).")

        print("  Computing metrics …")
        df = compute_all_metrics(images_dir)
        print_and_save_results(df, split.upper(), run_dir, args.dsname)

        if not df.empty:
            metric_cols = ["PSNR-L", "PSNR-C", "SSIM", "DeltaE76", "LPIPS"]
            means = df[metric_cols].mean()
            run_means[split] = {"run": run_name, **{k: round(v, 4) for k, v in means.items()}}

        print("  Applying gamma 2.2 to saved images …")
        apply_gamma_to_saved_images(images_dir)

        if args.images:
            print("  Building per-scene visualizations …")
            make_scene_visualizations(images_dir, display_gamma=2.2)

    return run_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra",      required=True)
    parser.add_argument("--run",        default=None)
    parser.add_argument("--checkpoint", default="best_val", choices=["best", "best_val", "last"])
    parser.add_argument("--dsname",     default="nikon")
    parser.add_argument("--skip_infer", action="store_true")
    parser.add_argument("--split",      default="both", choices=["test", "val", "both"])
    parser.add_argument("--images",     action="store_true")
    args = parser.parse_args()

    extra = args.extra.rstrip("/")

    if args.run:
        run_names = [args.run]
    else:
        selected  = pick_run_interactively(extra)
        run_names = selected if isinstance(selected, list) else [selected]

    summary = {}
    for run_name in run_names:
        run_means = evaluate_run(run_name, extra, args)
        for split, row in run_means.items():
            summary.setdefault(split, []).append(row)

    if len(run_names) > 1:
        runs_dir = os.path.join("./runs", extra)
        print(f"\n{'='*60}")
        print("  Summary across all runs:")
        for split, rows in summary.items():
            if not rows:
                continue
            df_summary = pd.DataFrame(rows)
            out = os.path.join(runs_dir, f"summary_{split}.csv")
            df_summary.to_csv(out, index=False)
            print(f"\n  {split.upper()} summary → {out}")
            print(df_summary.to_string(index=False))
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
