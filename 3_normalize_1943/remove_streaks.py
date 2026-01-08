import sys
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def read_grayscale_tif(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
    return arr, profile


def illumination_correction_gaussian(img, sigma):
    """ remove streaks with Gaussian smoothing """
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    result = np.zeros_like(img_norm)

    total_rows = img_norm.shape[0]
    chunk = 500  # process in row chunks 
    for start in tqdm(range(0, total_rows, chunk), desc="Processing rows"):
        end = min(start + chunk, total_rows)
        result[start:end, :] = gaussian_filter(img_norm[start:end, :], sigma=sigma, mode="nearest")

    background = (result - result.min()) / (result.max() - result.min())

    # subtract background
    corrected = img_norm - background
    corrected = (corrected - corrected.min()) / (corrected.max() - corrected.min())

    corrected = np.clip(corrected * 255, 0, 255)
    return corrected, background * 255


def main(input_path, output_path):
    img, profile = read_grayscale_tif(input_path)

    # correction
    corrected, background = illumination_correction_gaussian(img, sigma=200)

    out_dir = os.path.dirname(output_path) or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    profile.update(dtype="float32")

    corrected_path = output_path
    with rasterio.open(corrected_path, "w", **profile) as dst:
        dst.write(corrected.astype(np.float32), 1)

    # visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(background, cmap="gray")
    axes[1].set_title("Estimated Background (Gaussian)")
    axes[2].imshow(corrected, cmap="gray")
    axes[2].set_title("Corrected (After Illumination Removal)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    vis_path = os.path.join(out_dir, "streaks_comparison.png")
    plt.savefig(vis_path, dpi=150)
    plt.close()

    print(f"Saved corrected image -> {corrected_path}")
    print(f"Saved visualization -> {vis_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
