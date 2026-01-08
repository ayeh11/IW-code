import os
import json
import geopandas as gpd
import pandas as pd

def load_thresholds(thresholds_arg):
    # load thresholds from file
    if thresholds_arg.endswith(".csv"):
        df = pd.read_csv(thresholds_arg)
        if "Class" in df.columns and "OtsuThreshold_m2" in df.columns:
            return dict(zip(df["Class"], df["OtsuThreshold_m2"]))
    try:
        return json.loads(thresholds_arg)
    except json.JSONDecodeError:
        if os.path.exists(thresholds_arg):
            with open(thresholds_arg, "r") as f:
                return json.load(f)
        raise ValueError("Must be JSON string, JSON file, or CSV file.")

def filter_polygons(input_path, output_path, class_field, thresholds):
    # filter polygons by class-specific area thresholds and write output
    gdf = gpd.read_file(input_path)

    if gdf.crs is not None and gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:32733")

    gdf["area_m2"] = gdf.geometry.area.abs()

    kept = []
    dropped_summary = {}

    for cls, thr in thresholds.items():
        sub = gdf[gdf[class_field] == cls].copy()
        if sub.empty:
            continue
        keep_mask = sub["area_m2"] >= thr
        kept.append(sub[keep_mask])

        n_drop = (~keep_mask).sum()
        n_keep = keep_mask.sum()
        dropped_summary[cls] = {
            "threshold_m2": thr,
            "dropped": int(n_drop),
            "kept": int(n_keep),
            "dropped_%": round(100 * n_drop / len(sub), 2)
        }

    kept_gdf = gpd.GeoDataFrame(pd.concat(kept, ignore_index=True), crs=gdf.crs)
    kept_gdf.to_file(output_path)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--class_field", required=True)
    p.add_argument("--thresholds", required=True)
    args = p.parse_args()

    thresholds = load_thresholds(args.thresholds)
    filter_polygons(args.input, args.output, args.class_field, thresholds)
