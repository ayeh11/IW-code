import sys
import numpy as np
from skimage import exposure
import rasterio
import os

def read_grayscale_tif(path):
    # read a single-band tif 
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
    return arr, profile

def main(source_path, target_path, output_path):
    # match histogram of source to target and write normalized output
    source, src_profile = read_grayscale_tif(source_path)
    target, tgt_profile = read_grayscale_tif(target_path)

    # normalize to [0, 255]
    source = 255 * (source - source.min()) / (source.max() - source.min())
    target = 255 * (target - target.min()) / (target.max() - target.min())

    # outputs
    parent_dir = os.path.dirname(output_path) or "."
    os.makedirs(parent_dir, exist_ok=True)

    # matching
    matched = exposure.match_histograms(target, source)
    out_profile = tgt_profile.copy()
    out_profile.update(dtype="float32")
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(matched.astype(np.float32), 1)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
