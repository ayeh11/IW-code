import os
import argparse
import rasterio
from rasterio.windows import Window
import numpy as np

def tile_raster(input_path, output_dir, tile_size=1024, overlap=0, stride=None,
                nodata_override=None):
    # tile a raster into fixed-size tiles 
    os.makedirs(output_dir, exist_ok=True)

    if stride is None:
        stride = tile_size

    with rasterio.open(input_path) as src:
        W = src.width
        H = src.height
        count = src.count
        profile = src.profile.copy()

        nodata_vals = src.nodatavals if hasattr(src, 'nodatavals') else (src.nodata,)
        nodata = nodata_vals[0] if nodata_vals else src.nodata
        if nodata_override is not None:
            nodata = nodata_override

        row_starts = list(range(0, H, stride))
        col_starts = list(range(0, W, stride))

        n_written = 0
        for r in row_starts:
            for c in col_starts:
                win_h = min(tile_size, H - r)
                win_w = min(tile_size, W - c)
                if win_h <= 0 or win_w <= 0:
                    continue

                win = Window(c, r, win_w, win_h)

                data = src.read(window=win, masked=False)
                if nodata is not None:
                    try:
                        if (data == nodata).all():
                            continue
                    except Exception:
                        if np.all(data == nodata):
                            continue

                out_meta = profile.copy()
                out_meta.update({
                    'driver': 'GTiff',
                    'height': win_h,
                    'width': win_w,
                    'transform': src.window_transform(win),
                    'count': count,
                })
                if nodata is not None:
                    out_meta['nodata'] = nodata

                tile_name = f"tile_r{r:06d}_c{c:06d}.tif"
                out_path = os.path.join(output_dir, tile_name)

                out_meta['dtype'] = data.dtype.name

                with rasterio.open(out_path, 'w', **out_meta) as dst:
                    dst.write(data)

                n_written += 1



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output_dir', '-o', required=True)
    parser.add_argument('--tile_size', type=int, default=1024)
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--nodata', type=float, default=None)

    args = parser.parse_args()

    tile_raster(args.input, args.output_dir, tile_size=args.tile_size,
                overlap=args.overlap, stride=None, nodata_override=args.nodata)


if __name__ == '__main__':
    main()
