"""
In collaboration with Microsoft Corporation.
"""

import os
import math
import argparse
from collections import defaultdict
import shutil

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
import shapely.geometry as sgeom
import matplotlib.pyplot as plt
from tqdm import tqdm


def set_up_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--min_class_index", required=True, type=int)
    p.add_argument("--true_file_path", type=str, default=None, metavar="TIFF")
    p.add_argument("--pred_file_path", type=str, default=None, metavar="TIFF")
    p.add_argument("--metrics_dir_path", type=str, default=None)

    p.add_argument("--imagery_dir_path", type=str, default=None)
    p.add_argument("--mask_dir_path", type=str, default=None)
    p.add_argument("--predictions_dir_path", type=str, default=None)
    p.add_argument("--output_dir_path", type=str, default=None)
    p.add_argument("--metrics_subdir", type=str, default="metrics")

    p.add_argument("--imagery_year", type=str, default="")
    p.add_argument("--iou_threshold", type=float, default=0.5)
    p.add_argument("--polygone_metric", action="store_true")
    p.add_argument("--post_processing", action="store_true")
    p.add_argument("--std_range", type=float, default=1.0)
    p.add_argument("--sparsity_file_path", type=str, default=None)
    p.add_argument("--sparsity_threshold", type=float, default=None)
    p.add_argument("--models_dir_path", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p



def stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


# verify basenames match between gt and pred
def ensure_same_name_or_error(gt_path: str, pred_path: str):
    s_gt = stem(gt_path)
    s_pr = stem(pred_path)
    if s_pr == f"{s_gt}_split":
        return s_gt

    raise ValueError(f"Basenames must match (or pred may be <base>_split).\n  GT:   {s_gt}\n  Pred: {s_pr}")

# open raster and return array, transform, crs and shape
def open_raster(path: str):
    with rasterio.open(path) as src:
        arr = src.read(1)
        tfm = src.transform
        crs = src.crs
        shape = (src.height, src.width)
    return arr, tfm, crs, shape


# check that ground truth and prediction rasters align exactly
def check_alignment(gt_tfm, gt_crs, gt_shape, pr_tfm, pr_crs, pr_shape):
    same = (gt_shape == pr_shape and gt_crs == pr_crs and
        abs(gt_tfm.a - pr_tfm.a) < 1e-9 and
        abs(gt_tfm.e - pr_tfm.e) < 1e-9 and
        abs(gt_tfm.c - pr_tfm.c) < 1e-6 and
        abs(gt_tfm.f - pr_tfm.f) < 1e-6)
    if not same:
        raise ValueError("GT and prediction rasters must share grid/CRS/transform.")


# vectorize connected components for a given class id from raster mask
def gt_components_for_class(mask_arr: np.ndarray, transform, class_id: int):
    bin_arr = (mask_arr == class_id).astype(np.uint8)
    geoms = []
    for geom, val in shapes(bin_arr, mask=bin_arr.astype(bool), transform=transform):
        if val == 1:
            poly = sgeom.shape(geom)
            if not poly.is_empty:
                geoms.append(poly)
    return geoms


# vectorize predicted raster into polygons for classes >= min_class_index
def vectorize_pred_raster(pred_arr: np.ndarray, transform, min_class_index: int):
    classes = [int(c) for c in np.unique(pred_arr) if int(c) >= min_class_index]
    out = defaultdict(list) 
    for k in classes:
        bin_arr = (pred_arr == k).astype(np.uint8)
        for geom, val in shapes(bin_arr, mask=bin_arr.astype(bool), transform=transform):
            if val == 1:
                poly = sgeom.shape(geom)
                if not poly.is_empty:
                    out[k].append(poly)
    return out


# greedily match predicted and gt polygons by IoU threshold
def greedy_match_iou(pred_polys, gt_polys, iou_thr: float):
    if not pred_polys and not gt_polys:
        return 0, 0, 0, []
    if not pred_polys:
        return 0, 0, len(gt_polys), []
    if not gt_polys:
        return 0, len(pred_polys), 0, []

    p_areas = np.array([p.area for p in pred_polys])
    g_areas = np.array([g.area for g in gt_polys])
    ious = np.zeros((len(pred_polys), len(gt_polys)), dtype=float)
    for i, p in enumerate(pred_polys):
        for j, g in enumerate(gt_polys):
            inter = p.intersection(g).area
            if inter == 0.0:
                continue
            union = p_areas[i] + g_areas[j] - inter
            if union > 0:
                ious[i, j] = inter / union

    pairs = [(i, j, ious[i, j]) for i in range(len(pred_polys)) for j in range(len(gt_polys)) if ious[i, j] >= iou_thr]
    pairs.sort(key=lambda t: t[2], reverse=True)

    used_p, used_g, matches = set(), set(), []
    for i, j, v in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j, v))

    TP = len(matches)
    FP = len(pred_polys) - TP
    FN = len(gt_polys) - TP
    return TP, FP, FN, matches


# plot and save confusion matrix
def plot_conf_matrix(cnf: np.ndarray, class_names: list[str], title: str, out_png: str, normalize: bool):
    mat = cnf.astype(float).copy()
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = 100.0 * mat / row_sums

    plt.figure(figsize=(7, 7))
    plt.imshow(mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    thresh = (np.nanmax(mat) if mat.size else 1.0) / (1.5 if normalize else 2.0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            s = f"{val:.1f}" if normalize else f"{int(val)}"
            plt.text(j, i, s, ha="center", va="center", color="white" if val > thresh else "black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


# evaluate a single tile with metrics
def evaluate_single_tile(true_file_path: str,
                         pred_file_path: str,
                         metrics_dir_path: str,
                         min_class_index: int,
                         iou_threshold: float,
                         imagery_year: str):
    os.makedirs(metrics_dir_path, exist_ok=True)

    tile_id = ensure_same_name_or_error(true_file_path, pred_file_path)

    gt_arr, gt_tfm, gt_crs, gt_shape = open_raster(true_file_path)
    pr_arr, pr_tfm, pr_crs, pr_shape = open_raster(pred_file_path)
    check_alignment(gt_tfm, gt_crs, gt_shape, pr_tfm, pr_crs, pr_shape)

    min_k = max(0, int(min_class_index))
    gt_classes = sorted([int(c) for c in np.unique(gt_arr) if int(c) >= min_k])
    pred_polys_per_k = vectorize_pred_raster(pr_arr, pr_tfm, min_k)
    pred_classes = sorted(pred_polys_per_k.keys())
    eval_classes = sorted(set(gt_classes).union(set(pred_classes)))

    # confusion matrix
    max_class = max(eval_classes) if eval_classes else 0
    table_size = max_class + 1
    cnf = np.zeros((table_size, table_size), dtype=int)

    per_class = {}
    micro_TP = micro_FP = micro_FN = 0

    for k in eval_classes:
        gts_k = gt_components_for_class(gt_arr, gt_tfm, k)
        preds_k = pred_polys_per_k.get(k, [])
        TP, FP, FN, _ = greedy_match_iou(preds_k, gts_k, iou_threshold)

        cnf[k, k] += TP    # matches
        cnf[0, k] += FP    # false positives: predicted k, no GT
        cnf[k, 0] += FN    # false negatives: GT k, no pred

        P = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
        R = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
        F = 2 * P * R / (P + R) if (not math.isnan(P) and not math.isnan(R) and (P + R) > 0) else float("nan")
        per_class[k] = dict(TP=TP, FP=FP, FN=FN, Precision=P, Recall=R, F1=F)

        micro_TP += TP
        micro_FP += FP
        micro_FN += FN

    # summaries
    micro_P = micro_TP / (micro_TP + micro_FP) if (micro_TP + micro_FP) > 0 else float("nan")
    micro_R = micro_TP / (micro_TP + micro_FN) if (micro_TP + micro_FN) > 0 else float("nan")
    micro_F1 = 2 * micro_P * micro_R / (micro_P + micro_R) if (not math.isnan(micro_P) and not math.isnan(micro_R) and (micro_P + micro_R) > 0) else float("nan")

    macro_P = np.nanmean([per_class[k]["Precision"] for k in eval_classes]) if eval_classes else float("nan")
    macro_R = np.nanmean([per_class[k]["Recall"] for k in eval_classes]) if eval_classes else float("nan")
    macro_F1 = np.nanmean([per_class[k]["F1"] for k in eval_classes]) if eval_classes else float("nan")

    row = {"tile_id": tile_id,
           "micro_TP": micro_TP, "micro_FP": micro_FP, "micro_FN": micro_FN,
           "micro_P": micro_P, "micro_R": micro_R, "micro_F1": micro_F1,
           "macro_P": macro_P, "macro_R": macro_R, "macro_F1": macro_F1}
    for k in eval_classes:
        r = per_class[k]
        row.update({
            f"class_{k}_TP": r["TP"], f"class_{k}_FP": r["FP"], f"class_{k}_FN": r["FN"],
            f"class_{k}_Precision": r["Precision"], f"class_{k}_Recall": r["Recall"], f"class_{k}_F1": r["F1"]
        })

    base = os.path.join(metrics_dir_path, f"{tile_id}_polygone_metrics")

    # confusion matrices
    default_names = ["Negative", "Omuti", "BigTrees"]
    class_names = default_names[:table_size]
    title = f"Confusion matrix (Polygon): {imagery_year}".strip()
    png_raw = os.path.join(metrics_dir_path, f"{tile_id}_confusion_matrix_polygon_raw.png")
    png_norm = os.path.join(metrics_dir_path, f"{tile_id}_confusion_matrix_polygon_norm.png")
    plot_conf_matrix(cnf, class_names, title, png_raw, normalize=False)
    plot_conf_matrix(cnf, class_names, title, png_norm, normalize=True)


def get_imagery_files(imagery_dir, extensions=(".tif", ".tiff")):
    files = [os.path.join(imagery_dir, f) for f in os.listdir(imagery_dir)
             if f.lower().endswith(extensions)]
    files.sort()
    return files


def run_batch(args):
    # directories
    for dname, d in ("imagery_dir_path", args.imagery_dir_path), ("mask_dir_path", args.mask_dir_path), ("predictions_dir_path", args.predictions_dir_path):
        if not d or not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d} (argument --{dname})")

    out = args.output_dir_path
    if not out:
        raise FileNotFoundError("--output_dir_path is required for batch mode")

    if not os.path.isdir(out):
        os.makedirs(out, exist_ok=True)

    else:
        if args.overwrite:
            for entry in os.listdir(out):
                path = os.path.join(out, entry)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)


    imagery_files = get_imagery_files(args.imagery_dir_path)

    tile_metrics_paths = []
    tile_cnf_paths = []

    for image_path in tqdm(imagery_files, desc="Evaluating tiles"):
        base = stem(image_path)
        true_file = os.path.join(args.mask_dir_path, f"{base}.tif")
        pred_file = os.path.join(args.predictions_dir_path, f"{base}_split.tif")

        # tile output dirs
        tile_dir = os.path.join(args.output_dir_path, base)
        metrics_dir = os.path.join(tile_dir, args.metrics_subdir)
        os.makedirs(metrics_dir, exist_ok=True)

        # evaluate single tile
        evaluate_single_tile(
            true_file_path=true_file,
            pred_file_path=pred_file,
            metrics_dir_path=metrics_dir,
            min_class_index=args.min_class_index,
            iou_threshold=args.iou_threshold,
            imagery_year=args.imagery_year
        )

        tile_metrics_paths.append(os.path.join(metrics_dir, f"{base}_polygone_metrics.csv"))
        tile_cnf_paths.append(os.path.join(metrics_dir, f"{base}_confusion_matrix_polygon.csv"))

    combined = None
    label_order = ["Negative", "Omuti", "BigTrees"]

    valid_cnf = []
    for pth in tile_cnf_paths:
        if os.path.exists(pth):
            try:
                df = pd.read_csv(pth, index_col=0)
                # reindex to standard order (fill missing with 0)
                df = df.reindex(index=label_order, columns=label_order, fill_value=0)
                valid_cnf.append(df)
            except Exception as e:
                print(f"[WARN] Could not read cnf CSV: {pth} ({e})")

    if valid_cnf:
        combined_df = sum(valid_cnf)
        combined_dir = os.path.join(args.output_dir_path, "combined_metrics")
        os.makedirs(combined_dir, exist_ok=True)

        title = f"Combined Confusion Matrix (Polygon)"
        plot_conf_matrix(combined_df.values, label_order, title, os.path.join(combined_dir, "combined_confusion_counts.png"), normalize=False)
        plot_conf_matrix(combined_df.values, label_order, title, os.path.join(combined_dir, "combined_confusion_normalized.png"), normalize=True)

        metrics_rows = []
        total = combined_df.values.sum()
        for idx, cls_name in enumerate(label_order[1:], start=1):
            k = idx
            tp = combined_df.loc[label_order[k], label_order[k]]
            fp = combined_df.loc["Negative", label_order[k]]
            fn = combined_df.loc[label_order[k], "Negative"]
            P = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            R = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            F = 2 * P * R / (P + R) if (not np.isnan(P) and not np.isnan(R) and (P + R) > 0) else np.nan
            IoU = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
            # compute TN and class accuracy
            tn = total - (tp + fp + fn)
            Acc = (tp + tn) / total if total > 0 else np.nan
            metrics_rows.append(dict(Class=cls_name, Accuracy=Acc, IoU=IoU, Precision=P, Recall=R, F1=F))

        metrics_df = pd.DataFrame(metrics_rows).set_index("Class")
        metrics_df.loc["Mean"] = metrics_df.mean(numeric_only=True)

        # worst tiles summary
        worst_list = []
        for p in tile_metrics_paths:
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    # keep track of source CSV for traceability
                    df["source_metrics_csv"] = p
                    worst_list.append(df)
                except Exception as e:
                    print(f"[WARN] Could not read metrics CSV: {p} ({e})")

        if worst_list:
            all_tiles_df = pd.concat(worst_list, ignore_index=True, sort=False)
            ranked_base = os.path.join(combined_dir, "ranked_tiles")
            try:
                ranked_df = all_tiles_df.sort_values("micro_F1", ascending=False, na_position="last")
            except Exception:
                ranked_df = all_tiles_df
            ranked_df.to_csv(f"{ranked_base}.csv", index=False)


def main():
    args = set_up_parser().parse_args()

    # Choose mode
    batch_mode = (args.imagery_dir_path and args.mask_dir_path and args.predictions_dir_path and args.output_dir_path)

    if batch_mode:
        run_batch(args)
    else:
        if not (args.true_file_path and args.pred_file_path and args.metrics_dir_path):
            raise ValueError("Single-file mode requires --true_file_path, --pred_file_path, --metrics_dir_path.")
        evaluate_single_tile(
            true_file_path=args.true_file_path,
            pred_file_path=args.pred_file_path,
            metrics_dir_path=args.metrics_dir_path,
            min_class_index=args.min_class_index,
            iou_threshold=args.iou_threshold,
            imagery_year=args.imagery_year
        )


if __name__ == "__main__":
    main()
