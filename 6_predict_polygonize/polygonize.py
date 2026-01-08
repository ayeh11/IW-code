import argparse, os, csv
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt, gaussian_filter, label as ndi_label
from shapely.geometry import Point
from skimage.measure import label, regionprops
from tqdm import tqdm
from rasterio.merge import merge as merge_tifs


CLASS_ID_OMUTIS = 1
CLASS_ID_TREES = 2
COLOR_TREES = "#2ca02c"
COLOR_OMUTI = "#9467bd"

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def count_polygons(mask):
    # count number of polygons per target class in a mask
    row = {}

    # Omutis
    _, n_omuti = ndi_label(mask == CLASS_ID_OMUTIS)
    row["omuti_polygons"] = int(n_omuti)

    # BigTrees
    _, n_trees = ndi_label(mask == CLASS_ID_TREES)
    row["bigtrees_polygons"] = int(n_trees)

    return row

def append_csv(out_csv, row):
    # write to csv
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)

def px_area_m2_from_dataset(src):
    # compute pixel area in m2 
    tr = src.transform
    return abs(tr.a) * abs(tr.e)

def read_thresholds_csv(table_csv, class_name):
    # read area thresholds from a csv table
    df = pd.read_csv(table_csv)
    row = df[df["Class"].str.contains(class_name, case=False)]
    if row.empty:
        raise ValueError(f"{class_name} not found in {table_csv}")
    r = row.iloc[0]
    minv = float(r["OtsuThreshold_m2"])
    maxv = float(r.get("99thPercentile_m2", np.inf))
    return minv, maxv

def circle_overlap_ratio(a, b):
    # compute overlap ratio of two circles relative to smaller
    inter = a.intersection(b).area
    return inter / min(a.area, b.area)

def merge_overlapping_circles_soft(geoms,
                                   thresh_full=0.5,
                                   thresh_soft=0.25):
    # merge overlapping circle geometries using hard and soft thresholds
    changed = True
    merge_records = []
    while changed:
        changed = False
        new_geoms = []
        skip = set()
        for i, gi in enumerate(geoms):
            if i in skip:
                continue
            merged = gi
            ci = gi.centroid
            ri = np.sqrt(gi.area / np.pi)
            for j, gj in enumerate(geoms[i+1:], start=i+1):
                if j in skip:
                    continue
                cj = gj.centroid
                rj = np.sqrt(gj.area / np.pi)
                f = circle_overlap_ratio(gi, gj)
                if f > thresh_full:
                    merged = gi.union(gj)
                    merge_records.append((gi, gj, f, "full"))
                    skip.add(j)
                    changed = True
                elif thresh_soft <= f <= thresh_full:
                    xm, ym = (ci.x + cj.x)/2, (ci.y + cj.y)/2
                    d = np.hypot(ci.x - cj.x, ci.y - cj.y)
                    r_new = 0.5*d + max(ri, rj)
                    merged = Point(xm, ym).buffer(r_new)
                    merge_records.append((gi, gj, f, "soft"))
                    skip.add(j)
                    changed = True
            new_geoms.append(merged)
        geoms = new_geoms
    finals = []
    for g in geoms:
        c = g.centroid
        r = np.sqrt(g.area / np.pi)
        finals.append(Point(c.x, c.y).buffer(r))
    return finals, merge_records

def save_png(path, fig):
    # save figure as png 
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=150)
    plt.close(fig)

def draw_circles(ax, circles, color, lw=1):
    # draw circles on axis
    for g in circles:
        x, y = g.exterior.xy
        ax.plot(x, y, color=color, lw=lw, alpha=0.9)

def split_trees_circle(prob3, mask, px_area_m2, min_area_px,
                       prob_thr=0.3, min_dist=3):
    # detect tree centers from probability map and create circles
    region = prob3 > prob_thr
    if not region.any():
        return mask
    dt = distance_transform_edt(region)
    coords = peak_local_max(prob3, min_distance=min_dist,
                            threshold_abs=prob_thr, labels=region)
    if coords.size == 0:
        return mask

    circles = [Point(c[1], c[0]).buffer(dt[c[0], c[1]]) for c in coords]

    merged, merge_records = merge_overlapping_circles_soft(circles,
                                                           thresh_full=0.5,
                                                           thresh_soft=0.25)

    keep = [g for g in merged if g.area >= min_area_px * px_area_m2]

    refined = mask.copy()
    refined[mask == CLASS_ID_TREES] = 0
    rows, cols = np.indices(mask.shape)
    for g in keep:
        c = g.centroid
        r = np.sqrt(g.area / np.pi)
        rr = (cols - c.x)**2 + (rows - c.y)**2 <= (r**2)
        refined[rr] = CLASS_ID_TREES

    return refined

def filter_omutis(mask, px_area_m2, thresholds_csv, class_id=2, base_name="tile"):
    # filter omuti regions by area thresholds from table
    min_area_m2, max_area_m2 = read_thresholds_csv(thresholds_csv, "Omuti")
    omutis = (mask == class_id)
    if not omutis.any():
        return mask
    labeled = label(omutis)
    refined = mask.copy()
    kept = np.zeros_like(omutis)
    for region in regionprops(labeled):
        area_m2 = region.area * px_area_m2
        if min_area_m2 <= area_m2 <= max_area_m2:
            coords = tuple(zip(*region.coords))
            rows, cols = zip(*region.coords)
            refined[rows, cols] = class_id
            kept[rows, cols] = 1
        else:
            rows, cols = zip(*region.coords)
            refined[rows, cols] = 0

    return refined


def process_tile(in_tif, prob_npy, out_tif, min_area_table, prob_threshold, peak_min_distance):
    # process single tile
    with rasterio.open(in_tif) as src:
        mask = src.read(1)
        meta = src.meta.copy()
        px_area_m2 = px_area_m2_from_dataset(src)
    probs = np.load(prob_npy)
    prob3 = probs[CLASS_ID_TREES]
    prob3 = gaussian_filter(prob3, 0.8)

    out_path = Path(out_tif)

    min_area_trees, _ = read_thresholds_csv(min_area_table, "BigTree")
    min_area_px = max(1, int(round(min_area_trees / px_area_m2)))
    refined = split_trees_circle(prob3, mask, px_area_m2, min_area_px,
                                 prob_thr=prob_threshold,
                                 min_dist=peak_min_distance)
    refined = filter_omutis(refined, px_area_m2,
                            min_area_table,
                            class_id=CLASS_ID_OMUTIS,
                            base_name=out_path.stem)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta.update(dtype=rasterio.uint8, count=1, compress="lzw")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(refined.astype(np.uint8), 1)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-tif", required=True)
    p.add_argument("--prob-npy", required=True)
    p.add_argument("--out-tif", required=True)
    p.add_argument("--write-counts", action="store_true")
    p.add_argument("--counts-out", type=str)
    p.add_argument("--merged-out", type=str)
    p.add_argument("--min-area-table", required=True)
    p.add_argument("--prob-threshold", type=float, default=0.3)
    p.add_argument("--peak-min-distance", type=int, default=3)
    args = p.parse_args()

    in_tif_path = Path(args.in_tif)
    prob_path = Path(args.prob_npy)
    out_tif_path = Path(args.out_tif)

    if in_tif_path.is_dir():
        tif_files = sorted(in_tif_path.glob("*.tif"))

        processed_paths = []
        for tif in tqdm(tif_files, desc="Processing tiles", unit="tile"):
            base = tif.stem.replace("_predictions", "")
            npy_file = prob_path / f"{base}_probs.npy"
            if not npy_file.exists():
                continue
            out_file = out_tif_path / f"{base}_split.tif"
            try:
                out = process_tile(
                    in_tif=tif,
                    prob_npy=npy_file,
                    out_tif=out_file,
                    min_area_table=args.min_area_table,
                    prob_threshold=args.prob_threshold,
                    peak_min_distance=args.peak_min_distance
                )
                processed_paths.append(out)

                if args.write_counts:
                    with rasterio.open(out) as src_ref:
                        refined_mask = src_ref.read(1)

                    row = {"file": tif.name}
                    row.update(count_polygons(refined_mask))

                    append_csv(args.counts_out, row)
            except Exception as e:
                print(f"[ERROR] Failed on {tif.name}: {e}")

        if args.merged_out and processed_paths:
            try:
                srcs = [rasterio.open(str(p)) for p in processed_paths]
                mosaic, out_trans = merge_tifs(srcs)
                out_meta = srcs[0].meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "count": mosaic.shape[0],
                    "dtype": mosaic.dtype,
                })
                merged_path = Path(args.merged_out)
                merged_path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(merged_path, "w", **out_meta) as dest:
                    dest.write(mosaic)
                for s in srcs:
                    s.close()
            except Exception as e:
                print(f"[ERROR] Failed to merge cleaned tiles: {e}")


    else:
        # single tile mode
        out = process_tile(
            in_tif=in_tif_path,
            prob_npy=prob_path,
            out_tif=out_tif_path,
            min_area_table=args.min_area_table,
            prob_threshold=args.prob_threshold,
            peak_min_distance=args.peak_min_distance
        )

        if args.write_counts and out is not None:
            with rasterio.open(out) as src_ref:
                refined_mask = src_ref.read(1)
            row = {"file": Path(args.in_tif).name}
            row.update(count_polygons(refined_mask))
            append_csv(args.counts_out, row)


if __name__ == "__main__":
    main()
