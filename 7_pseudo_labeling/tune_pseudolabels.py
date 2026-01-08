import argparse, os, glob, time, sys
import numpy as np
from multiprocessing import Pool
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import matplotlib.pyplot as plt
from tqdm import tqdm
import cupy as cp

try:
    _gpu_available = cp.cuda.runtime.getDeviceCount() > 0
    if _gpu_available:
        xp = cp
        to_numpy = cp.asnumpy
        print("gpu detected")
    else:
        import numpy as np
        xp = np
        to_numpy = lambda x: x
        print("using cpu")
except Exception as e:
    xp = np
    to_numpy = lambda x: x
    print(f"using numpy")

CLASS_COL = "class_id"

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def vectorize_masks(mask_dir: str, background=0) -> gpd.GeoDataFrame:
    # vectorize mask tifs into polygons for analysis
    files = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
    if not files:
        raise SystemExit(f"No .tif masks found in {mask_dir}")

    geoms, class_ids, crs = [], [], None
    for f in files:
        with rasterio.open(f) as src:
            mask = src.read(1)
            transform = src.transform
            if crs is None:
                crs = src.crs
            for geom, val in shapes(mask, mask=mask != background, transform=transform):
                if int(val) == background:
                    continue
                geoms.append(shape(geom))
                class_ids.append(int(val))
    gdf = gpd.GeoDataFrame({CLASS_COL: class_ids, "geometry": geoms}, crs=crs)
    return gdf


def load_predictions(pred_dir: str) -> gpd.GeoDataFrame:
    # load predicted mask tifs and vectorize into geodataframe
    tif_files = sorted(glob.glob(os.path.join(pred_dir, "*.tif")))

    if tif_files:
        geoms, class_ids, crs = [], [], None
        for f in tif_files:
            with rasterio.open(f) as src:
                mask = src.read(1)
                transform = src.transform
                if crs is None:
                    crs = src.crs
                for geom, val in shapes(mask, mask=mask != 0, transform=transform):
                    if int(val) == 0:
                        continue
                    geoms.append(shape(geom))
                    class_ids.append(int(val))
        gdf = gpd.GeoDataFrame({"class_id": class_ids, "geometry": geoms}, crs=crs)
        return gdf

    raise SystemExit(f"[ERROR] No .tif or .gpkg predictions found in {pred_dir}")


def compute_empirical_pvalues(train_gdf, pred_gdf):
    # compute empirical p-values comparing predicted polygon areas to training
    if train_gdf.crs is None:
        raise SystemExit("Train polygons lack CRS.")
    if pred_gdf.crs != train_gdf.crs:
        pred_gdf = pred_gdf.to_crs(train_gdf.crs)

    train_gdf = train_gdf.copy()
    pred_gdf = pred_gdf.copy()
    train_gdf["area_m2"] = train_gdf.geometry.area
    pred_gdf["area_m2"] = pred_gdf.geometry.area

    classes = sorted(train_gdf[CLASS_COL].dropna().unique().tolist())
    train_areas = {c: xp.asarray(train_gdf.loc[train_gdf[CLASS_COL] == c, "area_m2"].values) for c in classes}

    pvals = xp.full(len(pred_gdf), xp.nan)
    iterator = pred_gdf.itertuples(index=False)

    for i, row in enumerate(iterator):
        c = getattr(row, CLASS_COL)
        A_pred = getattr(row, "geometry").area
        if c not in train_areas:
            continue
        A_train = train_areas[c]
        p = (1.0 + xp.sum(A_train >= A_pred)) / (len(A_train) + 1.0)
        pvals[i] = p

    pred_gdf["p_value"] = to_numpy(pvals)
    return pred_gdf


def polygon_iou(a, b):
    # compute intersection-over-union for two polygons
    inter = a.intersection(b).area
    if inter == 0:
        return 0.0
    uni = a.union(b).area
    return inter / uni if uni > 0 else 0.0


def score_against_val(kept_gdf, val_gdf, iou_thr=0.5):
    # score kept pseudo labels against validation polygons using IoU matching
    if kept_gdf.crs != val_gdf.crs:
        val_gdf = val_gdf.to_crs(kept_gdf.crs)
    classes = sorted(val_gdf[CLASS_COL].unique())
    rows = []

    preds_by_class = {}
    for c in classes:
        preds_c = kept_gdf[kept_gdf[CLASS_COL] == c]
        if len(preds_c):
            try:
                sidx = preds_c.sindex
            except Exception:
                sidx = None
        else:
            sidx = None
        preds_by_class[c] = (preds_c, sidx)

    for c in (tqdm(classes, desc="scoring classes") if len(classes) > 1 else classes):
        preds, sidx = preds_by_class[c]
        gts = val_gdf[val_gdf[CLASS_COL] == c]
        tp = 0
        if len(preds) == 0 or len(gts) == 0:
            fp = len(preds)
            fn = len(gts)
            prec = rec = f1 = 0
            rows.append({"class": c, "precision": prec, "recall": rec, "f1": f1})
            continue

        preds_r = preds.reset_index().rename(columns={"index": "pred_idx"})
        gts_r = gts.reset_index().rename(columns={"index": "gt_idx"})

        try:
            matches = gpd.sjoin(preds_r[["pred_idx", "geometry"]], gts_r[["gt_idx", "geometry"]],
                                how="inner", predicate="intersects")
        except Exception:
            matches = None

        if matches is None or matches.empty:
            fp = len(preds)
            fn = len(gts)
            prec = rec = f1 = 0
            rows.append({"class": c, "precision": prec, "recall": rec, "f1": f1})
            continue

        matches = matches.rename(columns={"index_right": "gt_row"}, errors="ignore")
        matches = matches.join(gts_r["geometry"].rename("gt_geometry"), on="gt_row")

        def _pair_iou(row):
            try:
                return polygon_iou(row["geometry"], row["gt_geometry"])
            except Exception:
                return 0.0

        matches["iou"] = matches.apply(_pair_iou, axis=1)
        matches_sorted = matches.sort_values("iou", ascending=False)

        matched_preds, matched_gts = set(), set()
        for _, r in matches_sorted.iterrows():
            pred_idx, gt_row, iou = r["pred_idx"], r["gt_row"], r["iou"]
            if iou < iou_thr or pred_idx in matched_preds or gt_row in matched_gts:
                continue
            matched_preds.add(pred_idx)
            matched_gts.add(gt_row)
            tp += 1

        fp = len(preds) - tp
        fn = len(gts) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        rows.append({"class": c, "precision": prec, "recall": rec, "f1": f1})

    df = pd.DataFrame(rows)
    macro_f1 = df["f1"].mean() if len(df) else 0
    return macro_f1, df


def plot_heatmap(df, value_col, out_png, title):
    # plot heatmap of ranges
    piv = df.pivot_table(index="lower", columns="upper", values=value_col)
    plt.imshow(piv.values, aspect="auto", origin="lower")
    plt.colorbar(label=value_col)
    plt.xticks(ticks=range(piv.shape[1]), labels=[f"{c:.2f}" for c in piv.columns], rotation=90)
    plt.yticks(ticks=range(piv.shape[0]), labels=[f"{r:.2f}" for r in piv.index])
    plt.title(title)
    plt.xlabel("upper")
    plt.ylabel("lower")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] wrote {out_png}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--train-mask-dir", required=True)
    ap.add_argument("--val-mask-dir", required=True)
    ap.add_argument("--tune-outdir", required=True)
    ap.add_argument("--lower-start", type=float, default=0.02)
    ap.add_argument("--lower-stop", type=float, default=0.12)
    ap.add_argument("--lower-step", type=float, default=0.01)
    ap.add_argument("--upper-start", type=float, default=0.88)
    ap.add_argument("--upper-stop", type=float, default=0.98)
    ap.add_argument("--upper-step", type=float, default=0.01)
    ap.add_argument("--require-all-classes", action="store_true")
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--objective", choices=["macro_f1", "max_total"], default="macro_f1")
    ap.add_argument("--num-workers", type=int, default=1)
    return ap.parse_args()


def evaluate_pair(args_tuple):
    lo, hi, pred_scored, val_gdf, iou_thr, require_all, classes = args_tuple
    t0 = time.time()
    kept = pred_scored[(pred_scored["p_value"] >= lo) & (pred_scored["p_value"] <= hi)]
    counts = kept.groupby(CLASS_COL).size() if len(kept) else pd.Series(dtype=int)
    per_class = {c: int(counts.get(c, 0)) for c in classes}
    total = int(len(kept))
    ok = (not require_all) or all(per_class[c] > 0 for c in classes)
    macro_f1, _ = score_against_val(kept, val_gdf, iou_thr)
    return {"lower": lo, "upper": hi, "total_kept": total, "ok": ok, "macro_f1": macro_f1,
            **{f"count_{c}": per_class[c] for c in classes}}


def main():
    args = parse_args()
    ensure_dir(args.tune_outdir)

    train_gdf = vectorize_masks(args.train_mask_dir)
    val_gdf = vectorize_masks(args.val_mask_dir)
    preds_val = load_predictions(args.pred_dir)

    pred_scored = compute_empirical_pvalues(train_gdf, preds_val)

    lowers = np.arange(args.lower_start, args.lower_stop + 1e-9, args.lower_step)
    uppers = np.arange(args.upper_start, args.upper_stop + 1e-9, args.upper_step)

    classes = sorted(val_gdf[CLASS_COL].unique())
    grid_args = [(lo, hi, pred_scored, val_gdf, args.iou_thresh,
                  args.require_all_classes, classes)
                 for lo in lowers for hi in uppers if lo < hi]

    rows = []
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            iterator = pool.imap_unordered(evaluate_pair, grid_args)
            for res in iterator:
                rows.append(res)
    else:
        for ga in grid_args:
            rows.append(evaluate_pair(ga))

    report = pd.DataFrame(rows)
    if len(report) == 0:
        print("empty grid")
        return

    if args.objective == "macro_f1":
        report = report.sort_values(["ok", "macro_f1", "total_kept"], ascending=[False, False, False])
    else:
        report = report.sort_values(["ok", "total_kept", "upper"], ascending=[False, False, True])

    best = report.iloc[0]
    lo, hi = best["lower"], best["upper"]
    f1 = best["macro_f1"]
    total = int(best["total_kept"])
    ok = bool(best["ok"])

    report_path = os.path.join(args.tune_outdir, "report.csv")
    best_summary = pd.DataFrame([{
        "lower": lo,
        "upper": hi,
        "total_kept": total,
        "macro_f1": f1,
        "ok": ok
    }])
    best_summary.to_csv(report_path, index=False)

    # heatmaps
    plot_heatmap(report, "macro_f1", os.path.join(args.tune_outdir, "heatmap_macro_f1.png"), "Macro-F1")
    plot_heatmap(report, "total_kept", os.path.join(args.tune_outdir, "heatmap_total_kept.png"), "Total Kept")


if __name__ == "__main__":
    main()
