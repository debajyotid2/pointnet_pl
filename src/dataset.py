"""Functions to transform and load training and test data."""
import os
import random
import logging

from typing import Any, Callable, Optional, Type
from urllib import request
from zipfile import ZipFile
from pathlib import Path

import numpy as np
import torch
import joblib

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.transforms import (
        Normalize, RandomRotationZAxis, AddRandomNoise, 
        PointToFloatTensor, PointToLongTensor 
)

torch.multiprocessing.set_sharing_strategy('file_system')

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)

class DownloadProgressBar(tqdm):
    """Progress bar to be displayed while downloading data from a URL"""
    def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

class ModelNet10(Dataset):
    """Dataset class for the ModelNet10 dataset"""
    _url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    __max_samples = 2048
    def __init__(self, 
                 root: str, 
                 download: bool = True, 
                 train: bool = True,
                 transform: Optional[Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any] | torch.Tensor]] = None,
                 target_transform: Optional[Callable[[np.ndarray[Any, Any] | torch.Tensor], torch.Tensor]] = None,
                 num_samples: int = 1024) -> None:
        super().__init__()

        self.train = train
        self.num_samples = num_samples
        
        filename = self._url.split("/")[-1]
        output_path = Path(root) / filename
        self._download(download, self._url, output_path)
        data_dirpath = self._extract_zip(output_path)
        file_list = self._generate_raw_file_list(data_dirpath, output_path.stem)
        self.class_name_dict = self._generate_class_name_dict(file_list)
        
        compressed_data_dir = data_dirpath / "compressed"
        self._compressed_file_list = self._sample_and_compress(file_list, compressed_data_dir)

        self.transform = transform
        self.target_transform = target_transform

    def _download(self, download: bool, url: str, output_path: Path) -> None:
        """
        Downloads the dataset to specified path. If dataset is not available on disk and 
        "download" is False, raises a RuntimeError.
        """
        if download and not output_path.with_name(output_path.stem).resolve().exists():
            self._download_url(url=self._url, output_path=output_path.resolve())
        elif not download and not output_path.with_name(output_path.stem).resolve().exists():
            raise RuntimeError(f"Dataset not found at {output_path.parent.resolve()}. Please set download=True to download the dataset.")
        
    def _download_url(self, url: str, output_path: Path) -> None:
        """Download a given URL with a progress bar."""
        if output_path.exists():
            return
        logging.info(f"Downloading from {url} to {output_path}.")
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as bar:
            request.urlretrieve(url, filename=output_path, reporthook=bar.update_to)

    def _extract_zip(self, zipfile_path: Path, data_dirpath: Optional[Path] = None) -> Path:
        """Extracts contents of a zip file given a zip file path, and returns path to extracted location."""
        if data_dirpath is None:
            data_dirpath = zipfile_path.resolve().with_name(zipfile_path.stem)
        if data_dirpath.exists():
            return data_dirpath
        data_dirpath.mkdir(exist_ok=True)
        with ZipFile(zipfile_path, "r") as zip_file:
            logging.info(f"Extracting to {data_dirpath}.")
            zip_file.extractall(path=data_dirpath)
        zipfile_path.unlink(missing_ok=True)

        return data_dirpath

    def _generate_raw_file_list(self, data_dirpath: Path, dataset_name: str) -> list[Path]:
        """Generate list of OFF files from dataset directory structure."""
        subset = "train" if self.train else "test"
        data_dirpath = data_dirpath / dataset_name
        pathlist = [list(class_dir.glob(subset)) for class_dir in data_dirpath.iterdir() if class_dir.is_dir()]
        filelist = [list(path[0].glob("*.off")) for path in pathlist]
        list_of_files = [file for files in filelist for file in files]
        return random.sample(list_of_files, k=len(list_of_files))
   
    def _generate_class_name_dict(self, file_list: list[Path]) -> dict[str, int]:
        """Generates dictionary containing class name to integer mappings"""
        classes = set(filepath.parents[1].name for filepath in file_list)
        return {class_name : label for class_name, label in zip(classes, range(len(classes)))}

    def _sample_and_compress(
            self,
            file_list: list[Path],
            compressed_data_dir: Path,
        ) -> list[Path]:
         
        def _sample_points_from_faces(mesh: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
            """Samples points uniformly from faces of a mesh, according to the face area."""

            def _calc_triangle_area(
                               vert1: np.ndarray[Any, Any],
                               vert2: np.ndarray[Any, Any],
                               vert3: np.ndarray[Any, Any]
                               ) -> np.ndarray[Any, Any]:
                """Calculate area of a triangle given its three vertices."""
                len_a = np.linalg.norm(vert1 - vert2)
                len_b = np.linalg.norm(vert2 - vert3)
                len_c = np.linalg.norm(vert3 - vert1)
                half_sum = 0.5 * (len_a + len_b + len_c)
                area = half_sum * (half_sum - len_a) * (half_sum - len_b) * (half_sum - len_c)
                return np.sqrt(np.maximum(area, 0))

            def _sample_from_face(
                               vert1: np.ndarray[Any, Any],
                               vert2: np.ndarray[Any, Any],
                               vert3: np.ndarray[Any, Any]
                               ) -> np.ndarray[Any, Any]:
               """
               Sample a single point given the vertices for a triangular face. Sampling is done using the
               formula for barycentric coordinates of a triangle.
               """ 
               randn_1, randn_2 = sorted([random.random(), random.random()])
               myfunc = lambda count: randn_1 * vert1[count] + (randn_2 - randn_1) * vert2[count] + (1 - randn_2) * vert3[count]
               pointsampler_numpy = np.frompyfunc(myfunc, nin=1, nout=1)
               return np.apply_along_axis(pointsampler_numpy, 0, np.arange(3))

            verts, faces = mesh
            area_calc_func = lambda face: _calc_triangle_area(verts[face[0]], verts[face[1]], verts[face[2]])
            sampling_func = lambda face: _sample_from_face(verts[face[0]], verts[face[1]], verts[face[2]])

            areas = list(map(area_calc_func, faces))

            sampled_faces = random.choices(faces, k=self.__max_samples, weights=areas)
            sampled_points = np.asarray(list(map(sampling_func, sampled_faces)), dtype=np.float32)

            return sampled_points

        def parse_off_points(off_filepath: Path) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
            """Parses 3-D coordinates and points in triangular faces from an OFF (Object File Format) file into numpy arrays."""
            skiprows = 0
            
            firstline = np.loadtxt(off_filepath, delimiter=",", dtype=np.str_, max_rows=1).item()
            
            if not firstline.startswith("OFF"):
                raise RuntimeError(f"File {off_filepath} is not a valid OFF file.")
            if firstline.strip() != "OFF":
                secondline = firstline.split("OFF")[1]
                n_verts_str, n_faces_str = secondline.strip().split(" ")[:2]
                n_verts, n_faces = int(n_verts_str), int(n_faces_str)
                skiprows = 1
            else:
                n_verts, n_faces, _ = np.loadtxt(off_filepath, dtype=np.int32, skiprows = 1, max_rows = 1)
                skiprows = 2
            return np.loadtxt(off_filepath, dtype=np.float32, skiprows=skiprows, max_rows=n_verts), \
                np.loadtxt(off_filepath, dtype=np.int32, usecols=[1, 2, 3], skiprows=skiprows+n_verts, max_rows=n_faces)


        def save_point_as_npz(npz_path: Path, point: np.ndarray[Any, Any]) -> None:
            """Saves a single data point to disk in the specified npz path."""
            if not npz_path.parent.exists():
                npz_path.parent.mkdir(parents=True)
            np.savez(npz_path, point=point)

        def _sample_and_compress_single_point(points_path: Path) -> Path:

            points_path = points_path.resolve()
            npz_path = compressed_data_dir / points_path.relative_to(points_path.parents[2])
            npz_path = npz_path.with_suffix(".npz")

            if npz_path.exists():
                return npz_path
            
            logging.debug(f"Compressing {points_path}")
            mesh = parse_off_points(points_path)
           
            points = _sample_points_from_faces(mesh)
            
            save_point_as_npz(npz_path, points)
            
            return npz_path

        if not compressed_data_dir.exists():
            compressed_data_dir.mkdir(parents=True)

        npz_paths = list()
        
        logging.info(f"Sampling and compressing {len(file_list)} points.")
        
        npz_paths = joblib.Parallel(n_jobs=-2)(
                joblib.delayed(_sample_and_compress_single_point)(file_path) for file_path in tqdm(file_list)
            )

        logging.info(f"Saved {len(npz_paths)} points to {compressed_data_dir}.")

        return npz_paths

    def __getitem__(self, idx: int | list[int] | torch.Tensor) ->\
            list[tuple[torch.Tensor | np.ndarray[Any, Any], int | torch.Tensor]] | \
            tuple[torch.Tensor | np.ndarray[Any, Any], int | torch.Tensor]:
        
        def get_class(file_path: Path) -> str:
            """Returns the class of an OFF file belonging to the dataset."""
            return file_path.parents[1].name

        def load_saved_npz_point(npz_path: Path) -> np.ndarray[Any, Any]:
            """Loads a single data point saved in a NumPy compressed format (.npz) from disk."""
            try:
                data = np.load(npz_path)
            except OSError:
                raise OSError(f"File {npz_path.resolve()} not found.")
            return data["point"]
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        points_paths = self._compressed_file_list[idx]
        if not type(points_paths) == list:
            points_paths = [points_paths]

        points = list(map(load_saved_npz_point, points_paths))
        targets = list(map(get_class, points_paths))
        

        data = [(point, self.class_name_dict[target]) for point, target in zip(points, targets)]

        if self.transform is not None:
            data = [(self.transform(point), target) for point, target in data]
        if self.target_transform is not None:
            data = [(point, self.target_transform(target)) for point, target in data]

        if len(data) == 1:
            return data[0]
        return data

    def __len__(self) -> int:
        if self._compressed_file_list:
            return len(self._compressed_file_list)
        return 0

class ModelNet40(ModelNet10):
    """Dataset class for the ModelNet40 dataset."""
    _url = "https://modelnet.cs.princeton.edu/ModelNet40.zip"

def train_val_split(data: torch.utils.data.Dataset, val_frac: float, seed: int = 42) -> \
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Splits a dataset into training and validation subsets."""
    num_val_samples = int(len(data) * val_frac)

    train_subset, val_subset = torch.utils.data.random_split(
            data, [len(data) - num_val_samples, num_val_samples], 
                   generator=torch.Generator().manual_seed(seed))
    return train_subset, val_subset

DATASET_DICT = {
            "ModelNet10": ModelNet10,
            "ModelNet40": ModelNet40
        }

def read_dataset(dataset: str) -> Type[Dataset]:
    """Returns the appropriate dataset class based on user input."""
    if dataset not in DATASET_DICT:
        raise ValueError(f"Invalid dataset type {dataset}. Must be one of {list(DATASET_DICT.keys())}")
    return DATASET_DICT[dataset]

def load_training_and_validation_data(
        batch_size: int,
        dataset: str = "ModelNet10",
        val_frac: float = 0.2, 
        augment: bool = True,
        num_workers: Optional[int] = None, 
        seed: int = 42) -> tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """Loads training and validation data into dataloaders and returns the dataloaders"""
    
    dataset_class = read_dataset(dataset)

    if augment:
        data_transform = transforms.Compose(
            [
                AddRandomNoise(),
                RandomRotationZAxis(),
                Normalize(),
                PointToFloatTensor()
            ]
        )
    else:
        data_transform = transforms.Compose(
            [
                Normalize(),
                PointToFloatTensor()
            ]
        )
    data = dataset_class(root=Path.cwd(), 
                            download=True, 
                            train=True, 
                            transform=data_transform, 
                            target_transform=PointToLongTensor())

    train, val = train_val_split(data, val_frac, seed)

    num_workers = os.cpu_count() - 1 if not num_workers else num_workers

    train_ds = DataLoader(train, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_ds = DataLoader(val, shuffle=True, batch_size=batch_size, num_workers=num_workers) if val else None

    return train_ds, val_ds

def load_test_data(batch_size: int, 
                   dataset: str = "ModelNet10", 
                   num_workers: Optional[int] = None) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Loads training data into a dataloader and returns the dataloader"""
    dataset_class = read_dataset(dataset)
    
    data_transform = transforms.Compose(
            [
                Normalize(),
                PointToFloatTensor()
            ]
        )
    data = dataset_class(
            root=Path.cwd(), 
            download=True, 
            train=False, 
            transform=data_transform, 
            target_transform=PointToLongTensor())

    num_workers = os.cpu_count() - 1 if not num_workers else num_workers
 
    return DataLoader(data, shuffle=False, batch_size=batch_size, num_workers=num_workers)


if __name__ == "__main__":
   train_ds, val_ds = load_training_and_validation_data(batch_size=3, dataset="ModelNet40")
   print(next(iter(train_ds)))
   print(next(iter(val_ds)))
