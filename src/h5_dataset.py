""" Functions to preprocess and load pointcloud data from the HDF5 dataset of ModelNet40. The 
dataset has to be manually downloaded from `https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip`.
The zip needs to be extracted to a directory. The directory path will be used to get a data generator using
functions in this module.
"""
import os
import logging

from typing import Any, Callable, Optional, Type
from pathlib import Path

import torch
import torchvision
import h5py
import numpy as np

from src.transforms import (Normalize, PointToLongTensor, PointToFloatTensor, AddRandomNoise, RandomRotationZAxis)
                            

TRAIN_H5_PART_SIZE = 2048 # Maximum number of data points in a single part HDF5 file of train subset of ModelNet40
TEST_H5_PART_SIZE = 2048  # Maximum number of data points in a single part HDF5 file of test subset of ModelNet40

DATASET_SIZE = {"train": 9840, "test": 2468}
TRAIN_FILES = ("ply_data_train0.h5", 
               "ply_data_train1.h5", 
               "ply_data_train2.h5", 
               "ply_data_train3.h5", 
               "ply_data_train4.h5") 
TEST_FILES = ("ply_data_test0.h5",
              "ply_data_test1.h5")

CLASS_NAME_DICT = { 
                    "airplane":	0, "bathtub":	1, "bed":	2, "bench":	3, "bookshelf":	4,
                    "bottle":	5, "bowl":	6, "car":	7, "chair":	8, "cone":	9, "cup": 10, 
                    "curtain":	11, "desk":	12, "door":	13, "dresser":	14, "flower_pot": 15, 
                    "glass_box":	16, "guitar":	17, "keyboard":	18, "lamp":	19,
                    "laptop":	20, "mantel":	21, "monitor":	22, "night_stand":	23,
                    "person":	24, "piano":	25, "plant":	26, "radio":	27,
                    "range_hood":	28, "sink":	29, "sofa":	30, "stairs":	31, "stool": 32, 
                    "table":	33, "tent":	34, "toilet":	35, "tv_stand":	36, "vase": 37, 
                    "wardrobe":	38, "xbox":	39 
                  }

def _get_class_name_dict() -> dict[str, int]:
    """Get a dictionary mapping class names to integer labels for ModelNet40."""
    return CLASS_NAME_DICT

def train_val_split(data: torch.utils.data.Dataset, val_frac: float, seed: int = 42) -> \
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Splits a dataset into training and validation subsets."""
    num_val_samples = int(len(data) * val_frac)

    train_subset, val_subset = torch.utils.data.random_split(
            data, [len(data) - num_val_samples, num_val_samples], 
                   generator=torch.Generator().manual_seed(seed))
    return train_subset, val_subset

class ModelNet40HDF5(torch.utils.data.Dataset):
    """Dataset class for the ModelNet40 HDF5 dataset"""
    def __init__(self, 
                 data_dirpath: Path,
                 train: bool = True,
                 transform: Optional[Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any] | torch.Tensor]] = None,
                 target_transform: Optional[Callable[[np.ndarray[Any, Any] | torch.Tensor], torch.Tensor]] = None,
                 num_samples: int = 1024) -> None:
        super().__init__()
        
        self._data_dirpath = data_dirpath

        self.subset = "train" if train else "test"
        self.num_samples = num_samples
        
        self.transform = transform
        self.target_transform = target_transform

        self.idxs = np.arange(self._get_dataset_size(self.subset))

    def _get_dataset_size(self, subset: str) -> int:
        """Returns the size of the subset of ModelNet40."""
        if subset not in DATASET_SIZE:
            raise ValueError(f"Subset must be one of {DATASET_SIZE.keys()}.")
        return DATASET_SIZE[subset]

    def __getitem__(self, idx: int | list[int] | torch.Tensor) ->\
            list[tuple[torch.Tensor | np.ndarray[Any, Any], int | torch.Tensor]] | \
            tuple[torch.Tensor | np.ndarray[Any, Any], int | torch.Tensor]:
        
        def _get_source_filename(subset: str, idx: int) -> str:
            """Returns name of HDF5 file from which the point is to be read."""
            if subset == "train":
                return TRAIN_FILES[idx // TRAIN_H5_PART_SIZE]
            else:
                return TEST_FILES[idx // TEST_H5_PART_SIZE]

        def _get_data_point(idx: int) \
                -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
            """Returns a datapoint retrieved from an HDF5 file corresponding to given subset of ModelNet10."""
            dataset_path = self._data_dirpath / _get_source_filename(self.subset, idx)
            logging.debug(f"Getting {idx} -> {idx % TRAIN_H5_PART_SIZE}")
            if self.subset == "train":
                idx = idx % TRAIN_H5_PART_SIZE - 1
            else:
                idx = idx % TEST_H5_PART_SIZE - 1
            with h5py.File(dataset_path, "r") as hdf:
                return hdf["data"][idx], hdf["label"][idx]

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not type(idx) == list:
            idx = [idx]

        data = list(map(_get_data_point, idx))

        rng = np.random.default_rng()
        data = [(rng.choice(point, size=self.num_samples, replace=False), target.item()) for point, target in data]

        if self.transform is not None:
            data = [(self.transform(point), target) for point, target in data]
        if self.target_transform is not None:
            data = [(point, self.target_transform(target)) for point, target in data]

        if len(data) == 1:
            return data[0]
        return data

    def __len__(self) -> int:
        if self.idxs is not None:
            return len(self.idxs)
        return 0

def load_training_and_validation_data(
        batch_size: int,
        data_dirpath: str = ".", 
        val_frac: float = 0.2, 
        num_samples: int = 1024,
        augment: bool = True,
        num_workers: Optional[int] = None, 
        seed: int = 42) -> tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader], dict[str, int]]:
    """Loads training and validation data into dataloaders and returns the dataloaders along with a dictionary containing classname to integer mappings.
    """
    data_dirpath = Path(data_dirpath)
    class_name_dict = _get_class_name_dict()
 
    if augment:
        data_transform = torchvision.transforms.Compose(
            [
                AddRandomNoise(),
                RandomRotationZAxis(),
                Normalize(),
                PointToFloatTensor()
            ]
        )
    else:
        data_transform = torchvision.transforms.Compose(
            [
                Normalize(),
                PointToFloatTensor()
            ]
        )
    data = ModelNet40HDF5(
                            data_dirpath=data_dirpath,
                            train=True,
                            num_samples=num_samples,
                            transform=data_transform, 
                            target_transform=PointToLongTensor()
                        )

    train, val = train_val_split(data, val_frac, seed)

    num_workers = os.cpu_count() - 1 if not num_workers else num_workers

    train_ds = torch.utils.data.DataLoader(train, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_ds = torch.utils.data.DataLoader(val, shuffle=True, batch_size=batch_size, num_workers=num_workers) if val else None

    return train_ds, val_ds, class_name_dict

def load_test_data(
        batch_size: int,
        data_dirpath: str = ".", 
        num_samples: int = 1024,
        num_workers: Optional[int] = None, 
        seed: int = 42) -> torch.utils.data.DataLoader:
    """
    Loads test data into a dataloader and returns the dataloader.
    """
    data_dirpath = Path(data_dirpath)
    
    data_transform = torchvision.transforms.Compose(
        [
            Normalize(),
            PointToFloatTensor()
        ]
    )
    data = ModelNet40HDF5(
                            data_dirpath=data_dirpath,
                            train=False,
                            num_samples=num_samples,
                            transform=data_transform, 
                            target_transform=PointToLongTensor()
                        )

    num_workers = os.cpu_count() - 1 if not num_workers else num_workers

    return torch.utils.data.DataLoader(data, shuffle=False, batch_size=batch_size, num_workers=num_workers)


if __name__ == "__main__":
    train_ds, val_ds, class_name_dict = load_training_and_validation_data(data_dirpath="./modelnet40_ply_hdf5_2048", batch_size=3)
    test_ds = load_test_data(data_dirpath="./modelnet40_ply_hdf5_2048", batch_size=3)
    train_points, train_labels = next(iter(train_ds))
    val_points, val_labels = next(iter(val_ds))
    test_points, test_labels = next(iter(test_ds))

    print(train_points.shape, train_labels.shape)
    print(train_labels)
    print(val_points.shape, val_labels.shape)
    print(test_points.shape, test_labels.shape)
