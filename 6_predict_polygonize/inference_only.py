"""
In collaboration with Microsoft Corporation.
"""

import argparse, os, time
from pathlib import Path
import numpy as np
import torch
import rasterio
from rasterio.merge import merge as rio_merge
from torchgeo.trainers import SemanticSegmentationTask
from tqdm import tqdm
import lightning.pytorch as pl  # noqa

LABELS = {"unlabeled": 0, "Omuti": 1, "BigTrees": 2}


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_model(ckpt, device):
    # load a trained semantic segmentation model from checkpoint
    model = SemanticSegmentationTask.load_from_checkpoint(
        ckpt, strict=False, map_location=device, weights=None
    )
    model.eval().to(device)
    return model


def read_tif_as_tensor(path, repeat_bands=0, force_channels=0):
    # read tif and normalize
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)
        meta = src.meta.copy()

    # normalize 
    if img.max() > 1:
        maxv = 65535 if img.max() > 255 else 255
        img = img / maxv

    c, h, w = img.shape

    if c == 1 and repeat_bands > 1:
        img = np.repeat(img, repeat_bands, axis=0)
        c = repeat_bands

    # fixed channels
    if force_channels > 0:
        if c > force_channels:
            img = img[:force_channels]
        elif c < force_channels:
            extra = np.repeat(img[-1:], force_channels - c, axis=0)
            img = np.concatenate([img, extra], axis=0)

    t = torch.from_numpy(img).unsqueeze(0)
    return t, meta


def save_mask(mask, meta, out):
    # save mask to tif
    ensure_dir(Path(out).parent)
    meta2 = meta.copy()
    meta2.update(count=1, dtype=rasterio.uint8, compress="lzw")

    with rasterio.open(out, "w", **meta2) as dst:
        dst.write(mask.astype(np.uint8), 1)


def save_probs(probs, meta, out_npy, out_tif=None):
    # save probability maps as npy
    ensure_dir(Path(out_npy).parent)
    np.save(out_npy, probs.astype(np.float32))

    if out_tif:
        k, h, w = probs.shape
        meta2 = meta.copy()
        meta2.update(count=k, dtype=rasterio.float32, compress="lzw")
        with rasterio.open(out_tif, "w", **meta2) as dst:
            dst.write(probs.astype(np.float32))


def build_combined_prediction(pred_tif_paths, out_path):
    # merge prediction tiles into a single mosaic tif
    if not pred_tif_paths:
        print("[WARN] No prediction tiles found; skipping mosaic.")
        return

    # Open rasters
    src_files = [rasterio.open(p) for p in pred_tif_paths]
    mosaic, out_transform = rio_merge(src_files)

    # Use metadata of first tile
    meta = src_files[0].meta.copy()
    meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "compress": "lzw",
        "count": 1,
        "dtype": rasterio.uint8,
    })

    ensure_dir(os.path.dirname(out_path))
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mosaic.astype(np.uint8))

    for s in src_files:
        s.close()


def predict_tile(model, device, img_path, outdir, save_probs_npy, save_probs_tif):
    # run model inference on a single image tile and save mask and probs
    tic = time.time()

    tensor, meta = read_tif_as_tensor(img_path)
    tensor = tensor.to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    mask = probs.argmax(axis=0).astype(np.uint8)

    base = img_path.stem
    pred_tif = f"{outdir}/predictions/{base}_predictions.tif"
    prob_npy = f"{outdir}/probmaps/{base}_probs.npy"
    prob_tif = f"{outdir}/probmaps/{base}_probs.tif" if save_probs_tif else None

    # Save outputs
    save_mask(mask, meta, pred_tif)
    if save_probs_npy or save_probs_tif:
        save_probs(probs, meta, prob_npy, prob_tif)

    return pred_tif


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--ckpt", required=True)
    a.add_argument("--input-dir", required=True)
    a.add_argument("--glob", default="*.tif")
    a.add_argument("--outdir", required=True)
    a.add_argument("--combined-pred", action="store_true")
    a.add_argument("--save-probs-npy", action="store_true")
    a.add_argument("--save-probs-tif", action="store_true")
    args = a.parse_args()

    # Device
    device = torch.device("cpu")

    # Load model
    model = load_model(args.ckpt, device)

    # Collect tiles
    input_dir = Path(args.input_dir)
    tif_paths = sorted(input_dir.glob(args.glob))
    if not tif_paths:
        raise ValueError(f"No TIFFs found in {input_dir} matching pattern {args.glob}")

    # Prepare output dirs
    ensure_dir(args.outdir)
    ensure_dir(Path(args.outdir) / "predictions")
    ensure_dir(Path(args.outdir) / "probmaps")

    # Process tiles
    pred_paths = []

    for p in tqdm(tif_paths, desc="Inference Progress", unit="tile"):
        pred_tif = predict_tile(
            model=model,
            device=device,
            img_path=p,
            outdir=args.outdir,
            save_probs_npy=args.save_probs_npy,
            save_probs_tif=args.save_probs_tif,
        )
        pred_paths.append(pred_tif)

    # Build combined mosaic if requested
    if args.combined_pred:
        build_combined_prediction(pred_paths, args.outdir + "/combined_predictions.tif")


if __name__ == "__main__":
    main()
