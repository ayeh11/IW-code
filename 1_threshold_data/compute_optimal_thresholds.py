import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import seaborn as sns
import pandas as pd

LABELS = ["Omuti", "BigTrees"]

def compute_otsu_threshold(areas):
    log_areas = np.log10(areas + 1e-6)
    t_log = threshold_otsu(log_areas)
    t_linear = 10 ** t_log
    return t_linear, t_log

def analyze_shapefile(path, class_field, outdir):
    os.makedirs(outdir, exist_ok=True)
    gdf = gpd.read_file(path)
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs("EPSG:32733")  # Namibia zone
    gdf["area_m2"] = gdf.geometry.area.abs()

    thresholds = []

    for c in LABELS:
        subset = gdf[gdf[class_field] == c]
        if subset.empty:
            print(f"[WARN] No polygons found for {c}")
            continue

        areas = subset["area_m2"].values
        t_linear, t_log = compute_otsu_threshold(areas)

        pct_below = (areas < t_linear).mean() * 100
        pct_above = 100 - pct_below

        max_area = np.max(areas)
        med_area = np.median(areas)
        mean_area = np.mean(areas)
        p99_area = np.percentile(areas, 99)

        thresholds.append({
            "Class": c,
            "OtsuThreshold_m2": round(t_linear, 2),
            "%Below": round(pct_below, 2),
            "%Above": round(pct_above, 2),
            "Median_m2": round(med_area, 2),
            "Mean_m2": round(mean_area, 2),
            "MaxArea_m2": round(max_area, 2),
            "99thPercentile_m2": round(p99_area, 2),
        })

        # plots
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(np.log10(areas + 1e-6), bins=40, stat="percent", fill=True,
                     color="#4A90E2", alpha=0.6)
        ax.axvline(t_log, color="red", linestyle="--", lw=2,
                   label=f"Otsu = {t_linear:.1f} m²")
        ax.axvline(np.log10(p99_area), color="green", linestyle=":", lw=1.5,
                   label=f"99th pct = {p99_area:.1f} m²")
        ax.set_title(f"{c} — log(Area) Distribution with Otsu")
        ax.set_xlabel("log(Area)")
        ax.set_ylabel("% of polygons")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{c}_otsu_threshold.png"), dpi=300)
        plt.close()

    # summary
    df = pd.DataFrame(thresholds)
    out_csv = os.path.join(outdir, "optimal_thresholds.csv")
    df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--shapefile", required=True)
    p.add_argument("--class_field", required=True)
    p.add_argument("--outdir", default="./analysis")
    args = p.parse_args()

    analyze_shapefile(args.shapefile, args.class_field, args.outdir)
