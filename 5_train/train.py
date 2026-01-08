"""
In collaboration with Microsoft Corporation.
"""

import argparse
import os
import json
import warnings
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchgeo.trainers import SemanticSegmentationTask
from datamodules import SegmentationDataModule

import rasterio
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")



def set_up_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--models_dir", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)

    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=3)


    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--min_epochs", type=int, default=10)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_batches_per_epoch", type=int, default=512)
    parser.add_argument("--val_batches_per_epoch", type=int, default=32)

    parser.add_argument("--early_stopping_patience", type=int, default=20)

    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use if CUDA is available.")
    parser.add_argument("--require_gpu", action="store_true",
                        help="If set, script exits with error unless CUDA GPU is available.")

    parser.add_argument("--loss", type=str, default="ce")
    parser.add_argument("--class_weights", action="store_true")
    parser.add_argument("--ignore_index", type=int, default=None)

    return parser



def compute_class_weights_from_tiles(data_dir):
    train_mask_dir = os.path.join(data_dir, "train", "mask")
    if not os.path.exists(train_mask_dir):
        raise ValueError(f"Training mask directory not found: {train_mask_dir}")

    class_counts = defaultdict(int)

    for mask_file in os.listdir(train_mask_dir):
        if mask_file.endswith(".tif"):
            mask_path = os.path.join(train_mask_dir, mask_file)
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                u, c = np.unique(mask, return_counts=True)
                for cls, ct in zip(u, c):
                    class_counts[cls] += ct

    total = sum(class_counts.values())
    num_classes = len(class_counts)
    weights = [total / (num_classes * class_counts[c]) for c in sorted(class_counts.keys())]

    return weights


def main(args):

    required_dirs = [
        "train/imagery", "train/mask",
        "val/imagery",  "val/mask",
        "test/imagery", "test/mask"
    ]
    for rd in required_dirs:
        p = os.path.join(args.data_dir, rd)
        if not os.path.exists(p):
            raise ValueError(f"Missing required directory: {p}")

    os.makedirs(args.models_dir, exist_ok=True)

    # gpu
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print("Using GPU:", torch.cuda.get_device_name(args.gpu_id))
    if args.require_gpu and not gpu_available:
        raise RuntimeError("--require_gpu flag set, but NO CUDA GPU detected.")
    device = torch.device(f"cuda:{args.gpu_id}" if gpu_available else "cpu")
    print("Selected device:", device)

    datamodule = SegmentationDataModule(
        base_data_dir=args.data_dir,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
        train_batches_per_epoch=args.train_batches_per_epoch,
        val_batches_per_epoch=args.val_batches_per_epoch,
    )

    class_weights = None
    if args.class_weights:
        class_weights = compute_class_weights_from_tiles(args.data_dir)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        with open(os.path.join(args.models_dir, "class_weights.json"), "w") as f:
            json.dump({"class_weights": class_weights.tolist()}, f, indent=2)

    task = SemanticSegmentationTask(
        model=args.model,
        backbone=args.backbone,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        loss=args.loss,
        class_weights=class_weights,
        ignore_index=args.ignore_index
    )


    task.to(device)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, mode="min"),
        ModelCheckpoint(dirpath=args.models_dir, save_top_k=1, save_last=True,
                        monitor="val_loss", filename="best-{epoch:02d}-{val_loss:.2f}")
    ]
    logger = TensorBoardLogger("logs/", name=args.experiment_name)

    trainer = pl.Trainer(
        accelerator="gpu" if gpu_available else "cpu",
        devices=1,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=args.models_dir,
        strategy="auto"
    )

    print("Starting training...")
    trainer.fit(task, datamodule=datamodule)

    print("Evaluating on test set...")
    results = trainer.test(task, datamodule=datamodule)

    with open(os.path.join(args.models_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(args.models_dir, "experiment_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    main(args)
