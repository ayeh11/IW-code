import os, math, json, argparse, random
from collections import defaultdict
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize, shapes as rio_shapes
from rasterio.windows import Window
from rasterio.transform import Affine
import geopandas as gpd
from tqdm import tqdm


LABELS = {
    "unlabeled": 0,
    "Omuti": 1,
    "BigTrees": 2,
}
ID2NAME = {v: k for k, v in LABELS.items()}


def shapefile_to_mask(shapefile_path, imagery_path, output_mask_path, class_field, all_touched=True):
    ''' make masks from labels '''
    with rasterio.open(imagery_path) as src:
        transform, crs, height, width = src.transform, src.crs, src.height, src.width
        dtype = "uint8"

    gdf = gpd.read_file(shapefile_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    series = gdf[class_field]
    try:
        is_numeric = pd.api.types.is_numeric_dtype(series.dtype)
    except Exception:
        is_numeric = False

    if is_numeric:
        gdf["mapped_class"] = series.fillna(0).astype(int)
    else:
        gdf["mapped_class"] = series.map(LABELS).fillna(0).astype(int)
    shapes = ((geom, val) for geom, val in zip(gdf.geometry, gdf["mapped_class"]))

    mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        fill=0,
        transform=transform,
        all_touched=all_touched,
        dtype=dtype,
    )

    with rasterio.open(
        output_mask_path, "w",
        driver="GTiff",
        height=height, width=width, count=1, dtype=dtype,
        crs=crs, transform=transform
    ) as dst:
        dst.write(mask, 1)

    with rasterio.open(output_mask_path) as chk:
        arr = chk.read(1)
        uniques, counts = np.unique(arr, return_counts=True)

    return output_mask_path


def _count_polygons_from_mask(mask_arr):
    poly_counts = defaultdict(int)
    try:
        for _, val in rio_shapes(mask_arr, mask=mask_arr != 0, transform=Affine.identity()):
            v = int(val)
            if v > 0:
                poly_counts[v] += 1
    except Exception:
        pass
    return dict(poly_counts)

def _tile_dirs(base_dir):
    for split in ("train", "val", "test"):
        for sub in ("imagery", "mask"):
            os.makedirs(os.path.join(base_dir, split, sub), exist_ok=True)

def ids_to_names(d, include_background=False):
    out = {}
    for k, v in d.items():
        cid = int(k)
        if cid == 0 and not include_background:
            continue
        out[ID2NAME.get(cid, f"class_{cid}")] = int(v)
    return out


def tile_and_split(imagery_path, mask_path, out_dir, tile_size=1024, splits=(80,10,10), seed=42):
    ''' tile labeled data and split into train/val/test '''
    random.seed(seed); np.random.seed(seed)

    _tile_dirs(out_dir)

    with rasterio.open(imagery_path) as src_img, rasterio.open(mask_path) as src_msk:
        assert src_img.crs == src_msk.crs and src_img.transform == src_msk.transform, \
            "Imagery and mask must be aligned (CRS & transform)."

        W, H = src_img.width, src_img.height
        step = tile_size  # no overlap in this simplified version
        nx = math.ceil(W / step)
        ny = math.ceil(H / step)

        mask_full = src_msk.read(1)
        uniques, counts = np.unique(mask_full, return_counts=True)

        # collect all candidate tiles with metadata
        candidates = []
        for i in tqdm(range(ny), desc="Scanning tiles"):
            for j in range(nx):
                x0, y0 = j * step, i * step
                window = Window(x0, y0, tile_size, tile_size).intersection(Window(0,0,W,H))

                r0, c0 = int(window.row_off), int(window.col_off)
                r1, c1 = int(window.row_off + window.height), int(window.col_off + window.width)
                mwin = mask_full[r0:r1, c0:c1]
                if mwin.size == 0 or not np.any(mwin > 0):
                    continue  # skip empty mask tiles

                # read data
                img_win = src_img.read(window=window)
                msk_win = src_msk.read(1, window=window)
                win_transform = src_img.window_transform(window)

                # compute per-tile counts
                uniq, cnts = np.unique(msk_win, return_counts=True)
                class_pixels = {int(u): int(c) for u, c in zip(uniq, cnts)}
                labeled_pixels = int(sum(cnts[uniq > 0]))
                poly_counts = _count_polygons_from_mask(msk_win)

                candidates.append({
                    "i": int(i), "j": int(j),
                    "window": window,
                    "transform": win_transform,
                    "img": img_win,
                    "msk": msk_win,
                    "class_pixels": class_pixels,
                    "poly_counts": poly_counts,
                    "labeled_pixels": labeled_pixels,
                    "total_pixels": int(msk_win.size),
                })

        if not candidates:
            raise RuntimeError("No tiles with labeled pixels found.")

        # group by class-combo signature (ignoring 0)
        groups = defaultdict(list)
        for t in candidates:
            present = tuple(sorted([c for c in t["class_pixels"].keys() if c > 0]))
            groups[present].append(t)

        # normalize splits
        ssum = sum(splits)
        train_p, val_p, test_p = [s/ssum for s in splits]

        split_bins = {"train": [], "val": [], "test": []}

        for combo, tiles in groups.items():
            random.shuffle(tiles)
            n = len(tiles)
            n_train = int(round(n * train_p))
            n_val = int(round(n * val_p))
            # ensure total counts exactly n
            n_train = min(n_train, n) 
            n_val   = min(n_val, n - n_train)
            n_test  = n - n_train - n_val

            split_bins["train"].extend(tiles[:n_train])
            split_bins["val"].extend(tiles[n_train:n_train+n_val])
            split_bins["test"].extend(tiles[n_train+n_val:])

        # write tiles directly into split folders + manifests
        manifests = {
            "train": [],
            "val": [],
            "test": [],
        }

        for split, tiles in split_bins.items():
            img_dir = os.path.join(out_dir, split, "imagery")
            msk_dir = os.path.join(out_dir, split, "mask")

            for idx, t in enumerate(tiles):
                name = f"tile_{t['i']:04d}_{t['j']:04d}.tif"
                img_path = os.path.join(img_dir, name)
                msk_path = os.path.join(msk_dir, name)

                # write imagery
                with rasterio.open(imagery_path) as src_img_template:
                    meta = src_img_template.meta.copy()
                meta.update({
                    "height": t["window"].height,
                    "width": t["window"].width,
                    "transform": t["transform"],
                })
                with rasterio.open(img_path, "w", **meta) as dst:
                    dst.write(t["img"])

                # write mask (single band)
                with rasterio.open(mask_path) as src_msk_template:
                    mmeta = src_msk_template.meta.copy()
                mmeta.update({
                    "height": t["window"].height,
                    "width": t["window"].width,
                    "transform": t["transform"],
                    "count": 1,
                    "dtype": t["msk"].dtype,
                })
                with rasterio.open(msk_path, "w", **mmeta) as dst:
                    dst.write(t["msk"], 1)

                manifests[split].append({
                    "tile": name,
                    "imagery_path": img_path,
                    "mask_path": msk_path,
                    "labeled_pixels": t["labeled_pixels"],
                    "total_pixels": t["total_pixels"],
                    "class_pixels_json": json.dumps(ids_to_names(t["class_pixels"]), sort_keys=True),
                    "class_polygons_json": json.dumps(ids_to_names(t["poly_counts"]), sort_keys=True),
                })

        id2name = {v:k for k,v in LABELS.items()}
        class_id_list = [cid for cid in sorted(id2name.keys()) if cid != 0]
        class_name_list = [id2name[cid] for cid in class_id_list]

        for split, rows in manifests.items():
            df = pd.DataFrame(rows)
            out_csv = os.path.join(out_dir, f"{split}.csv")
            df.to_csv(out_csv, index=False)

            # summary 
            tot_name = defaultdict(int)
            poly_name = defaultdict(int)
            for r in rows:
                cp = json.loads(r.get("class_pixels_json") or "{}")
                pp = json.loads(r.get("class_polygons_json") or "{}")
                for name, v in cp.items():
                    try:
                        tot_name[str(name)] += int(v)
                    except Exception:
                        pass
                for name, v in pp.items():
                    try:
                        poly_name[str(name)] += int(v)
                    except Exception:
                        pass

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapefile_path", required=True, type=str)
    p.add_argument("--class_field", required=True, type=str)
    p.add_argument("--imagery_path", required=True, type=str)
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--tile_size", type=int, default=1024)
    p.add_argument("--splits", type=int, nargs=3, default=[80,10,10])
    p.add_argument("--all_touched", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mask_path = os.path.join(args.output_dir, "mask_full.tif")

    shapefile_to_mask(
        shapefile_path=args.shapefile_path,
        imagery_path=args.imagery_path,
        output_mask_path=mask_path,
        class_field=args.class_field,
        all_touched=args.all_touched
    )

    tile_and_split(
        imagery_path=args.imagery_path,
        mask_path=mask_path,
        out_dir=args.output_dir,
        tile_size=args.tile_size,
        splits=tuple(args.splits),
        seed=42
    )

if __name__ == "__main__":
    main()
