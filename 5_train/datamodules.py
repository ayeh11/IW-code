"""
In collaboration with Microsoft Corporation.
"""

import os
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import RandomBatchGeoSampler
from torchgeo.transforms import AugmentationSequential
from torchgeo.datasets.utils import BoundingBox
from lightning.pytorch import LightningDataModule
import kornia.augmentation as K
import matplotlib.pyplot as plt
import torch
import numpy as np

def _remove_bboxes(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "bbox":
                continue
            if isinstance(v, BoundingBox):
                continue
            out[k] = _remove_bboxes(v)
        return out
    elif isinstance(obj, (list, tuple)):
        return [_remove_bboxes(v) for v in obj]
    else:
        return obj

# Preprocessing function to be applied to each sample during training
def preprocess(sample):
    if "image" in sample:
        sample["image"] = (sample["image"] / 255.0).float()
    if "mask" in sample:
        m = sample["mask"].squeeze().long()
        m[m == 4] = 0        
        sample["mask"] = m

    sample = _remove_bboxes(sample)
    return sample

# Preprocess for validation and testing
def preprocess2(sample):
    if "image" in sample:
        sample["image"] = (sample["image"] / 255.0).float()

    if "mask" in sample:
        m = sample["mask"]
        if m.ndim == 3 and m.shape[0] == 1:
            m = m.squeeze(0)
        else:
            m = m.squeeze()
        m = m.long()
        m[m == 4] = 0 
        sample["mask"] = m

    sample = _remove_bboxes(sample)
    return sample


# SingleRasterDataset class to load a single raster dataset during inference
class SingleRasterDataset(RasterDataset):
    def __init__(self, fn, transforms=None):
        self.filename_regex = os.path.basename(fn)
        super().__init__(root=os.path.dirname(fn), transforms=transforms)


class SegmentationDataModule(LightningDataModule):

    def __init__(
        self,
        base_data_dir,
        batch_size=64,
        patch_size=256,
        num_workers=6,
        train_batches_per_epoch=512,
        val_batches_per_epoch=32,
    ):
        super().__init__()

        self.base_data_dir = base_data_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.train_batches_per_epoch = train_batches_per_epoch
        self.val_batches_per_epoch = val_batches_per_epoch

        # paths
        self.train_imagery_path = os.path.join(base_data_dir, "train", "imagery")
        self.train_mask_path = os.path.join(base_data_dir, "train", "mask")
        self.val_imagery_path = os.path.join(base_data_dir, "val", "imagery")
        self.val_mask_path = os.path.join(base_data_dir, "val", "mask")
        self.test_imagery_path = os.path.join(base_data_dir, "test", "imagery")
        self.test_mask_path = os.path.join(base_data_dir, "test", "mask")

        for path in [self.train_imagery_path, self.train_mask_path, 
                    self.val_imagery_path, self.val_mask_path,
                    self.test_imagery_path, self.test_mask_path]:
            if not os.path.exists(path):
                raise ValueError(f"Required path does not exist: {path}")

        # add augmentation step
        self.train_augs = AugmentationSequential(
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(p=0.5, degrees=45),
            K.RandomRotation(p=0.5, degrees=135),
            K.RandomRotation(p=0.5, degrees=225),
            data_keys=["image", "mask"],
        )

    def setup(self, stage=None):
        # train datasets
        self.train_image_ds = RasterDataset(
            self.train_imagery_path,
            transforms=preprocess
        )
        self.train_mask_ds = RasterDataset(
            self.train_mask_path,
            transforms=preprocess
        )
        self.train_mask_ds.is_image = False
        self.train_ds = self.train_image_ds & self.train_mask_ds

        # validation datasets
        self.val_image_ds = RasterDataset(
            self.val_imagery_path,
            transforms=preprocess2
        )
        self.val_mask_ds = RasterDataset(
            self.val_mask_path,
            transforms=preprocess2
        )
        self.val_mask_ds.is_image = False
        self.val_ds = self.val_image_ds & self.val_mask_ds

        # test datasets
        self.test_image_ds = RasterDataset(
            self.test_imagery_path,
            transforms=preprocess2
        )
        self.test_mask_ds = RasterDataset(
            self.test_mask_path,
            transforms=preprocess2
        )
        self.test_mask_ds.is_image = False
        self.test_ds = self.test_image_ds & self.test_mask_ds

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Apply batch augmentations to the batch after it is transferred to the device."""
        if self.trainer:
            if self.trainer.training:
                batch = self.train_augs(batch)
        return batch

    def train_dataloader(self):
        sampler = RandomBatchGeoSampler(
            self.train_ds, 
            size=self.patch_size, 
            batch_size=self.batch_size, 
            length=self.train_batches_per_epoch * self.batch_size
        )

        return DataLoader(
            self.train_ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples
        )

    def val_dataloader(self):
        sampler = RandomBatchGeoSampler(
            self.val_ds, 
            size=self.patch_size, 
            batch_size=self.batch_size, 
            length=self.val_batches_per_epoch * self.batch_size
        )

        return DataLoader(
            self.val_ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples
        )

    def test_dataloader(self):
        sampler = RandomBatchGeoSampler(
            self.test_ds, 
            size=self.patch_size, 
            batch_size=self.batch_size, 
            length=self.val_batches_per_epoch * self.batch_size
        )

        return DataLoader(
            self.test_ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples
        )

    def plot(self, input_sample):
        fig = plt.figure()
        return fig