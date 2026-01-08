import argparse
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask

os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def crop_tif_to_shp_bbox(tif_path, shp_path, output_path):
    shapefile = gpd.read_file(shp_path)
    if shapefile.empty:
        raise ValueError("empty")
    if shapefile.crs is None:
        raise ValueError("no CRS")

    with rasterio.open(tif_path) as src:
        if shapefile.crs != src.crs:
            shapefile = shapefile.to_crs(src.crs)

        minx, miny, maxx, maxy = shapefile.total_bounds
        rb = src.bounds
        minx, miny = max(minx, rb.left), max(miny, rb.bottom)
        maxx, maxy = min(maxx, rb.right), min(maxy, rb.top)

        window = from_bounds(minx, miny, maxx, maxy, src.transform).round_offsets().round_lengths()
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

        data = src.read(window=window)
        transform = src.window_transform(window)

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": window.height,
            "width": window.width,
            "transform": transform
        })

        ensure_dir(os.path.dirname(output_path))
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(data)


def save_ring_crop(tif_path, shp_path, ring_output_path, ring_buffer, nodata_value=None):
    """buffer zone"""
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("empty")
    if gdf.crs is None:
        raise ValueError("no CRS")

    with rasterio.open(tif_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        minx, miny, maxx, maxy = gdf.total_bounds
        rb = src.bounds
        outer = box(
            max(minx - ring_buffer, rb.left),
            max(miny - ring_buffer, rb.bottom),
            min(maxx + ring_buffer, rb.right),
            min(maxy + ring_buffer, rb.top)
        )

        ob_minx, ob_miny, ob_maxx, ob_maxy = outer.bounds
        window = from_bounds(ob_minx, ob_miny, ob_maxx, ob_maxy, src.transform)
        window = window.round_offsets().round_lengths()
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

        data = src.read(window=window)
        transform = src.window_transform(window)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": int(window.height),
            "width": int(window.width),
            "transform": transform
        })

        nodata = nodata_value if nodata_value is not None else getattr(src, 'nodata', None)
        if nodata is None:
            if out_meta.get('dtype', '').startswith('uint') or out_meta.get('dtype', '') == 'uint8':
                nodata = 0
            else:
                nodata = 0

        inner = box(minx, miny, maxx, maxy)
        mask = geometry_mask([inner], transform=transform, invert=True, out_shape=(int(window.height), int(window.width)))

        if mask.any():
            for b in range(data.shape[0]):
                band = data[b]
                band[mask] = nodata
                data[b] = band

        ensure_dir(os.path.dirname(ring_output_path))
        out_meta['nodata'] = nodata
        with rasterio.open(ring_output_path, "w", **out_meta) as dst:
            dst.write(data)

    return outer.bounds


def save_combined_crop(tif_path, shp_path, ring_buffer, output_path):
    """buffer zone and original"""
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("empty")
    if gdf.crs is None:
        raise ValueError("no CRS")

    with rasterio.open(tif_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        minx, miny, maxx, maxy = gdf.total_bounds
        rb = src.bounds
        full_minx = max(minx - ring_buffer, rb.left)
        full_miny = max(miny - ring_buffer, rb.bottom)
        full_maxx = min(maxx + ring_buffer, rb.right)
        full_maxy = min(maxy + ring_buffer, rb.top)

        window = from_bounds(full_minx, full_miny, full_maxx, full_maxy, src.transform)
        window = window.round_offsets().round_lengths()
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

        data = src.read(window=window)
        transform = src.window_transform(window)

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": window.height,
            "width": window.width,
            "transform": transform
        })

        ensure_dir(os.path.dirname(output_path))
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif_path", required=True)
    parser.add_argument("--shp_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--ring_output_path", required=False)
    parser.add_argument("--post_pseudo_output_path", required=False)
    parser.add_argument("--ring_buffer", type=float, required=False)
    args = parser.parse_args()

    crop_tif_to_shp_bbox(args.tif_path, args.shp_path, args.output_path)

    if args.ring_output_path and args.post_pseudo_output_path and args.ring_buffer:
        save_ring_crop(args.tif_path, args.shp_path, args.ring_output_path, args.ring_buffer)
        save_combined_crop(args.tif_path, args.shp_path, args.ring_buffer, args.post_pseudo_output_path)


if __name__ == "__main__":
    main()
