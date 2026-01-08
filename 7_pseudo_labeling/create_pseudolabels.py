import argparse
import os
import glob
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

CLASS_COL = "class_id"

ID2NAME = {
    0: "unlabeled",
    1: "Omuti",
    2: "BigTrees",
}

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def vectorize_tif_masks(mask_dir: str, background=0) -> gpd.GeoDataFrame:
    tif_files = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
    if not tif_files:
        raise SystemExit(f"[ERROR] No .tif training masks found in {mask_dir}")

    geoms, class_ids, crs = [], [], None

    for f in tif_files:
        with rasterio.open(f) as src:
            mask = src.read(1)
            transform = src.transform
            if crs is None:
                crs = src.crs

            for geom, val in shapes(mask, mask=mask != background, transform=transform):
                val = int(val)
                if val == background:
                    continue
                geoms.append(shape(geom))
                class_ids.append(val)

    gdf = gpd.GeoDataFrame(
        {CLASS_COL: class_ids, "geometry": geoms},
        crs=crs
    )
    return gdf

def load_predictions(pred_dir: str) -> gpd.GeoDataFrame:
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

        gdf = gpd.GeoDataFrame(
            {CLASS_COL: class_ids, "geometry": geoms},
            crs=crs
        )
        return gdf

    raise SystemExit(f"[ERROR] No .tif or .gpkg prediction files found in {pred_dir}")


def compute_empirical_pvalues(train_gdf: gpd.GeoDataFrame, pred_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if train_gdf.crs is None:
        raise SystemExit("Train GPKG has no CRS; cannot compute areas meaningfully.")
    if pred_gdf.crs != train_gdf.crs:
        pred_gdf = pred_gdf.to_crs(train_gdf.crs)

    train_gdf = train_gdf.copy()
    pred_gdf = pred_gdf.copy()
    train_gdf["area_m2"] = train_gdf.geometry.area
    pred_gdf["area_m2"] = pred_gdf.geometry.area

    classes = sorted(train_gdf[CLASS_COL].dropna().unique().tolist())
    train_areas = {c: train_gdf.loc[train_gdf[CLASS_COL] == c, "area_m2"].values for c in classes}

    pvals = np.full(len(pred_gdf), np.nan)
    for i, row in pred_gdf.iterrows():
        c = row[CLASS_COL]
        A_pred = row["area_m2"]
        A_train = train_areas.get(c, [])
        if len(A_train) == 0:
            continue
        p = (1.0 + np.sum(A_train >= A_pred)) / (len(A_train) + 1.0)
        pvals[i] = p

    pred_gdf["p_value"] = pvals
    return pred_gdf


def apply_bounds(pred_scored: gpd.GeoDataFrame, lower: float, upper: float) -> gpd.GeoDataFrame:
    kept = pred_scored[(pred_scored["p_value"] >= lower) & (pred_scored["p_value"] <= upper)].copy()
    return kept[[CLASS_COL, "p_value", "geometry"]]


def write_filtered(gdf: gpd.GeoDataFrame, out_gpkg: str) -> str:
    ensure_dir(os.path.dirname(out_gpkg))
    _, ext = os.path.splitext(out_gpkg)
    ext = ext.lower()
    if ext == ".shp":
        driver = "ESRI Shapefile"
    else:
        driver = "GPKG"

    gdf_out = gdf.copy()
    if CLASS_COL in gdf_out.columns:
        def _id_to_name(v):
            try:
                return ID2NAME.get(int(v), f"class_{int(v)}")
            except Exception:
                return str(v)
        gdf_out["Name"] = gdf_out[CLASS_COL].apply(_id_to_name)

    gdf_out.to_file(out_gpkg, driver=driver)
    return out_gpkg


def merge_train_and_pseudo(train_gdf: gpd.GeoDataFrame, pseudo_path: str, merge_out: str) -> str:

    pseudo = gpd.read_file(pseudo_path)
    if pseudo.crs != train_gdf.crs:
        pseudo = pseudo.to_crs(train_gdf.crs)

    merged = gpd.GeoDataFrame(
        pd.concat([train_gdf[[CLASS_COL, "geometry"]], pseudo[[CLASS_COL, "geometry"]]],
                  ignore_index=True),
        geometry="geometry", crs=train_gdf.crs
    )

    def _id_to_name(v):
        try:
            return ID2NAME.get(int(v), f"class_{int(v)}")
        except Exception:
            return str(v)

    merged["Name"] = merged[CLASS_COL].apply(_id_to_name)

    ensure_dir(os.path.dirname(merge_out))
    _, ext = os.path.splitext(merge_out)
    driver = "ESRI Shapefile" if ext.lower() == ".shp" else "GPKG"

    merged.to_file(merge_out, driver=driver)
    return merge_out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-gpkg", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-gpkg", required=True)
    ap.add_argument("--merge-out", default=None)
    ap.add_argument("--pval-lower", type=float, required=True)
    ap.add_argument("--pval-upper", type=float, required=True)
    ap.add_argument("--out-format", choices=["gpkg", "shp"], default=None)
    return ap.parse_args()


def main():
    args = parse_args()

    train_gdf = vectorize_tif_masks(args.train_gpkg)
    if CLASS_COL not in train_gdf.columns:
        raise SystemExit(f"Training GPKG missing '{CLASS_COL}' column")

    pred_gdf = load_predictions(args.pred_dir)
    pred_scored = compute_empirical_pvalues(train_gdf, pred_gdf)
    kept = apply_bounds(pred_scored, args.pval_lower, args.pval_upper)

    out_path = args.out_gpkg
    if args.out_format:
        fmt = args.out_format.lower()
        if fmt == "shp":
            out_path = os.path.splitext(out_path)[0] + ".shp"
        else:
            out_path = os.path.splitext(out_path)[0] + ".gpkg"

    write_filtered(kept, out_path)

    if args.merge_out:
        merge_out_path = args.merge_out
        if args.out_format:
            fmt = args.out_format.lower()
            if fmt == "shp":
                merge_out_path = os.path.splitext(merge_out_path)[0] + ".shp"
            else:
                merge_out_path = os.path.splitext(merge_out_path)[0] + ".gpkg"
        merge_train_and_pseudo(train_gdf, out_path, merge_out_path)


if __name__ == "__main__":
    main()
