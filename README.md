# Deep Learning for Archival Aerial Photography: Understanding Long-Term Socioenvironmental Change in Namibia

This repository contains code and utilities for preparing geospatial imagery and labels, training semantic segmentation models, running inference, post-processing, and evaluation for my IW.

Quick layout
- `0_original_data/` — raw imagery and original shapefiles
- `1_threshold_data/` — compute and apply size thresholds to filter small polygons
- `2_crop_data/` — crop large imagery to area of interest defined by labels
- `3_normalize_1943/` — remove streaks and normalize 1943 imagery to 1970 distribution
- `4a_tile_labeled_data/` — create labeled tiles (train/val/test)
- `4b_tile_unlabeled_data/` — create unlabeled tiles for pseudo-labeling (rings)
- `5_train/` — training scripts and datamodules
- `6_predict_polygonize/` — inference and polygonization utilities
- `7_pseudo_labeling/` — tune and create pseudo labels
- `8_train_predict_evaluate/` — evaluation and run-through of entire process on combined datasets
- `info/` — debugging outputs and analysis

Requirements & creating the environment
- Create conda environments for training and all other uses

```bash
# Linux for gpu
conda env create -f IW_train.yml 

# macOS / zsh
conda env create -f IW_geo.yml 
conda env create -f IW_ml_mac.yml 

conda activate IW_geo
```

Each step by folder:

1) 1_threshold_data — compute & filter small polygons

- Compute optimal thresholds per class (plots + CSV):

```bash
python 1_threshold_data/compute_optimal_thresholds.py \
  --shapefile ../0_original_data/1970_data/1970_labels.shp \
  --class_field Name \
  --outdir ./1_threshold_data/1970_data/

python 1_threshold_data/compute_optimal_thresholds.py \
  --shapefile ../0_original_data/1943_data/1943_labels.shp \
  --class_field Name \
  --outdir ./1_threshold_data/1943_data/
```

- Filter small polygons using the CSV of thresholds:

```bash
python 1_threshold_data/filter_small_polygons.py \
  --input ../0_original_data/1970_data/1970_labels.shp \
  --output ./1_threshold_data/1970_data/1970_labels_filtered.shp \
  --class_field Name \
  --thresholds ./1_threshold_data/1970_data/optimal_thresholds.csv

python 1_threshold_data/filter_small_polygons.py \
    --input ../0_original_data/1943_data/1943_labels.shp \
    --output ./1_threshold_data/1943_data/1943_labels_filtered.shp \
    --class_field Name \
    --thresholds ./1_threshold_data/1943_data/optimal_thresholds.csv
```

2) 2_crop_data — crop imagery to label extents

- Crop imagery and create rings for pseudo-labeling (example):

```bash
python 2_crop_data/cropping.py \
  --tif_path ../0_original_data/1970_data/Aerial1970.tif \
  --shp_path ../1_threshold_data/1970_data/1970_labels_filtered.shp \
  --output_path ./2_crop_data/1970_data/cropped_Aerial1970.tif \
  --ring_output_path ./2_crop_data/1970_data/ring_Aerial1970.tif \
  --post_pseudo_output_path ./2_crop_data/1970_data/Aerial1970_post_pseudo.tif \
  --ring_buffer 2048

python 2_crop_data/cropping.py \
  --tif_path ../0_original_data/1943_data/Aerial1943.tif \
  --shp_path ../1_threshold_data/1943_data/1943_labels_filtered.shp \
  --output_path ./2_crop_data/1943_data/cropped_Aerial1943.tif
```

3) 3_normalize_1943 — remove streaks and normalize 1943 to 1970

- Remove streaks then normalize colors to match 1970 distribution:

```bash
python 3_normalize_1943/remove_streaks.py \
  ../2_crop_data/1943_data/cropped_Aerial1943.tif \
  ./3_normalize_1943/outputs/Aerial_1943_illuminated.tif

python 3_normalize_1943/normalize_1943.py \
  ../2_crop_data/1970_data/cropped_Aerial1970.tif \
  ./3_normalize_1943/outputs/Aerial_1943_illuminated.tif \
  ./3_normalize_1943/outputs/Aerial_1943_norm.tif
```

4) 4a_tile_labeled_data — prepare labeled tiles (train/val/test)

- Create labeled tiles (examples):

```bash
python 4a_tile_labeled_data/prepare_labeled_tiles.py \
  --shapefile_path ../1_threshold_data/1970_data/1970_labels_filtered.shp \
  --class_field Name \
  --imagery_path ../2_crop_data/1970_data/cropped_Aerial1970.tif \
  --output_dir ./4a_tile_labeled_data/1970_tiles/ \
  --tile_size 1024 \
  --splits 80 10 10

python 4a_tile_labeled_data/prepare_labeled_tiles.py \
  --shapefile_path ../1_threshold_data/1943_data/1943_labels_filtered.shp \
  --class_field Name \
  --imagery_path ../3_normalize_1943/outputs/Aerial_1943_norm.tif \
  --output_dir ./4a_tile_labeled_data/1943_tiles/ \
  --tile_size 1024 \
  --splits 80 10 10
```

5) 4b_tile_unlabeled_data — prepare unlabeled/pseudo tiles

- Prepare unlabeled tiles from the ring output (for pseudo labeling):

```bash
python 4b_tile_unlabeled_data/prepare_unlabeled_tiles.py \
  --input ../2_crop_data/1970_data/ring_Aerial1970.tif \
  --output_dir ./4b_tile_unlabeled_data/1970_tiles_pseudo \
  --tile_size 1024
```

6) 5_train — training

Done in IW through slurm to Della clusters for GPU computation 
- Train baseline model for 1970 and 1943:

```bash
conda activate IW_train

python3 5_train/train.py \
  --data_dir ./4a_tile_labeled_data/1970_tiles/ \
  --models_dir ./5_train/models/baseline \
  --experiment_name baseline \
  --class_weights \
  --require_gpu

python3 5_train/train.py \
  --data_dir ./4a_tile_labeled_data/1943_tiles/ \
  --models_dir ./5_train/models/baseline_1943 \
  --experiment_name baseline_1943 \
  --class_weights \
  --require_gpu
```

Notes:
- `--class_weights` computes inverse-frequency weights from training masks and saves them under `models/`.
- `--require_gpu` will cause the script to fail early if no CUDA GPU is available (useful in multi-host scripts when GPU availability is required).

7) 6_predict_polygonize — inference + polygonize

- Predict on tiles and produce polygon outputs:
  - For 1970: predict on test data, ring for pseudolabels, and 1970s validation data for tuning
  - For 1943: predict just on test tiles 

```bash
conda activate IW_ml_mac

python ./6_predict_polygonize/inference_only.py \
    --ckpt ./5_train/models/baseline/last.ckpt \
    --input-dir ./4b_tile_unlabeled_data/1970_tiles_pseudo/ \
    --outdir ./6_predict_polygonize/predictions/baseline_pseudo \
    --save-probs-npy \
    --save-probs-tif \
    --combined-pred

python ./6_predict_polygonize/inference_only.py \
    --ckpt ./5_train/models/baseline/last.ckpt \
    --input-dir ./4a_tile_labeled_data/1970_tiles/val/imagery \
    --outdir ./6_predict_polygonize/predictions/baseline_val \
    --save-probs-npy \
    --save-probs-tif \
    --combined-pred

python ./6_predict_polygonize/inference_only.py \
    --ckpt ./5_train/models/baseline/last.ckpt \
    --input-dir ./4a_tile_labeled_data/1970_tiles/test/imagery \
    --outdir ./6_predict_polygonize/predictions/baseline_test \
    --save-probs-npy \
    --save-probs-tif \
    --combined-pred

python ./6_predict_polygonize/inference_only.py \
    --ckpt ./5_train/models/baseline_1943/last.ckpt \
    --input-dir ./4a_tile_labeled_data/1943_tiles/test/imagery \
    --outdir ./6_predict_polygonize/predictions/baseline_1943 \
    --save-probs-npy \
    --save-probs-tif \
    --combined-pred

conda activate IW_geo

python ./6_predict_polygonize/polygonize.py \
  --in-tif ./6_predict_polygonize/predictions/baseline_pseudo/predictions/ \
  --prob-npy ./6_predict_polygonize/predictions/baseline_pseudo/probmaps/ \
  --out-tif ./6_predict_polygonize/predictions/baseline_pseudo/cleaned/ \
  --min-area-table ./1_threshold_data/1970_data/optimal_thresholds.csv \
  --prob-threshold 0.30 \
  --peak-min-distance 3 \
  --write-counts \
  --counts-out ./6_predict_polygonize/predictions/baseline_pseudo/pred_counts.csv 

python ./6_predict_polygonize/polygonize.py \
  --in-tif ./6_predict_polygonize/predictions/baseline_val/predictions/ \
  --prob-npy ./6_predict_polygonize/predictions/baseline_val/probmaps/ \
  --out-tif ./6_predict_polygonize/predictions/baseline_val/cleaned/ \
  --min-area-table ./1_threshold_data/1970_data/optimal_thresholds.csv \
  --prob-threshold 0.30 \
  --peak-min-distance 3 \
  --write-counts \
  --counts-out ./6_predict_polygonize/predictions/baseline_val/pred_counts.csv 

python ./6_predict_polygonize/polygonize.py \
  --in-tif ./6_predict_polygonize/predictions/baseline_test/predictions/ \
  --prob-npy ./6_predict_polygonize/predictions/baseline_test/probmaps/ \
  --out-tif ./6_predict_polygonize/predictions/baseline_test/cleaned/ \
  --min-area-table ./1_threshold_data/1970_data/optimal_thresholds.csv \
  --prob-threshold 0.30 \
  --peak-min-distance 3 \
  --write-counts \
  --counts-out ./6_predict_polygonize/predictions/baseline_test/pred_counts.csv 

python ./6_predict_polygonize/polygonize.py \
  --in-tif ./6_predict_polygonize/predictions/baseline_1943/predictions/ \
  --prob-npy ./6_predict_polygonize/predictions/baseline_1943/probmaps/ \
  --out-tif ./6_predict_polygonize/predictions/baseline_1943/cleaned/ \
  --min-area-table ./1_threshold_data/1943_data/optimal_thresholds.csv \
  --prob-threshold 0.30 \
  --peak-min-distance 3 \
  --write-counts \
  --counts-out ./6_predict_polygonize/predictions/baseline_1943/pred_counts.csv 
```

8) 7_pseudo_labeling — tune and create pseudo labels

- Tune thresholds for pseudo labeling:

```bash
python 7_pseudo_labeling/tune_pseudolabels.py \
  --pred-dir ../6_predict_polygonize/predictions/baseline_pseudo/polygons \
  --train-mask-dir ../4a_tile_labeled_data/1970_tiles/train/mask \
  --val-mask-dir ../4a_tile_labeled_data/1970_tiles/val/mask \
  --tune-outdir ./7_pseudo_labeling/tuning/ \
  --lower-start 0.02 --lower-stop 0.12 --lower-step 0.01 \
  --upper-start 0.88 --upper-stop 0.98 --upper-step 0.01 \
  --require-all-classes --objective macro_f1 --iou-thresh 0.5 --num-workers 4 --verbose

python 7_pseudo_labeling/create_pseudolabels.py \
  --pred-dir ../6_predict_polygonize/predictions/baseline_pseudo/polygons \
  --out-gpkg ./7_pseudo_labeling/pseudo_labels/filtered_ring_1970.shp \
  --pval-lower 0.01 --pval-upper 0.56 \
  --merge-out ./7_pseudo_labeling/all_labels/train_plus_pseudo_1970.shp
```

9) 8_train_predict_evaluate — end-to-end run (train on pseudo-labeled data, predict, evaluate)

- Example workflow includes preparing tiles from merged labels, training, predicting, polygonizing and evaluating. See `8_train_predict_evaluate/instructions.txt` for detailed example commands.

