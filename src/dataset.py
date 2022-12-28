"""Functions to transform and load training and test data. Some of the transformation 
classes have been adapted from https://colab.research.google.com/drive/1K_RsM3db8bPrXsIcxV7Qv4cHJa-M2xSn"""

import os
import random

from typing import Any, Callable, Optional
from urllib import request
from zipfile import ZipFile
from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

torch.multiprocessing.set_sharing_strategy('file_system')

class DownloadProgressBar(tqdm):
    """Progress bar to be displayed while downloading data from a URL"""
    def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

class ModelNet10(Dataset):
    """Dataset class for the ModelNet10 dataset"""
    def __init__(self, 
                 root: str, 
                 download: bool = True, 
                 train: bool = True,
                 transform: Optional[Callable[[np.ndarray[Any, Any] | torch.Tensor], torch.Tensor]] = None,
                 target_transform: Optional[Callable[[np.ndarray[Any, Any] | torch.Tensor], torch.Tensor]] = None,
                 sample: Optional[bool] = True,
                 num_samples: int = 1024) -> None:
        super().__init__()

        self.train = train
        self.sample = sample
        self.num_samples = num_samples
        
        url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
        filename = url.split("/")[-1]
        output_path = Path(root) / filename
        if download and not output_path.with_name(output_path.stem).resolve().exists():
            self._download_url(url=url, output_path=output_path.resolve())
        data_dirpath = self._extract_zip(output_path)

        self._file_list = self._generate_file_list(data_dirpath)
        self.class_name_dict = self._generate_class_name_dict()

        self.transform = transform
        self.target_transform = target_transform
        
    def _download_url(self, url: str, output_path: Path) -> None:
        """Download a given URL with a progress bar."""
        if output_path.exists():
            return
        print(f"Downloading from {url} to {output_path}.")
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
            print(f"Extracting to {data_dirpath}.")
            zip_file.extractall(path=data_dirpath)
        zipfile_path.unlink(missing_ok=True)

        return data_dirpath


    def _generate_file_list(self, data_dirpath: Path, dataset_name: str = "ModelNet10") -> list[Path]:
        """Generate list of OFF files from dataset directory structure."""
        subset = "train" if self.train else "test"
        data_dirpath = data_dirpath / dataset_name
        pathlist = [list(class_dir.glob(subset)) for class_dir in data_dirpath.iterdir() if class_dir.is_dir()]
        filelist = [list(path[0].glob("*.off")) for path in pathlist]
        list_of_files = [file for files in filelist for file in files]
        return random.sample(list_of_files, k=len(list_of_files))
   
    def _generate_class_name_dict(self) -> dict[str, int]:
        """Generates dictionary containing class name to integer mappings"""
        classes = set(filepath.parents[1].name for filepath in self._file_list)
        return {class_name : label for class_name, label in zip(classes, range(len(classes)))}
    
    def _sample_points_from_faces(self, mesh: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        """Samples points uniformly from faces of a mesh, according to the face area."""
        def _calc_triangle_area(
                           vert1: np.ndarray[Any, Any],
                           vert2: np.ndarray[Any, Any],
                           vert3: np.ndarray[Any, Any]
                           ) -> float:
            """Calculate area of a triangle given its three vertices."""
            len_a = np.linalg.norm(vert1 - vert2)
            len_b = np.linalg.norm(vert2 - vert3)
            len_c = np.linalg.norm(vert3 - vert1)
            half_sum = 0.5 * (len_a + len_b + len_c)
            area = half_sum * (half_sum - len_a) * (half_sum - len_b) * (half_sum - len_c)
            return max(area, 0) ** 0.5

        def _sample_from_face(
                           vert1: np.ndarray[Any, Any],
                           vert2: np.ndarray[Any, Any],
                           vert3: np.ndarray[Any, Any]
                           ) -> tuple[float, float, float]:
           """
           Sample a single point given the vertices for a triangular face. Sampling is done using the
           formula for barycentric coordinates of a triangle.
           """ 
           randn_1, randn_2 = sorted([random.random(), random.random()])
           myfunc = lambda count: randn_1 * vert1[count] + (randn_2 - randn_1) * vert2[count] + (1 - randn_2) * vert3[count]
           return (myfunc(0), myfunc(1), myfunc(2))

        verts, faces = mesh
        
        area_calc_func = lambda face: _calc_triangle_area(verts[face[0]], verts[face[1]], verts[face[2]])
        sampling_func = lambda face: _sample_from_face(verts[face[0]], verts[face[1]], verts[face[2]])

        areas = list(map(area_calc_func, faces))
        sampled_faces = (random.choices(faces, weights=areas, cum_weights=None, k=self.num_samples))
        
        sampled_points = np.asarray(list(map(sampling_func, sampled_faces)), dtype=np.float32)

        return sampled_points

    def __getitem__(self, idx: int | list[int] | torch.Tensor) -> torch.Tensor:
        def parse_off_points(off_filepath: Path) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
            """Parses 3-D coordinates and points in triangular faces from an OFF (Object File Format) file into numpy arrays."""
            def parse_single_point(line: str) -> tuple[float, float, float]:
                """Parse coordinates of a single vertex from a line of the OFF file."""
                coords = line.strip().split(" ")
                return (float(coords[0]), float(coords[1]), float(coords[2]))
            
            def parse_single_face(line: str) -> tuple[int, int, int]:
                """Parse vertice of a single face from a line of the OFF file."""
                points = line.strip().split(" ")
                return (int(points[0]), int(points[1]), int(points[2]))

            with open(off_filepath, "r") as file:
                if file.readline().strip() != "OFF":
                    raise RuntimeError(f"File {off_filepath} is not a valid OFF file.")
                n_verts_str, n_faces_str = file.readline().strip().split(" ")[:2]
                n_verts, n_faces = int(n_verts_str), int(n_faces_str)
                return np.asarray(list(map(
                    parse_single_point, [file.readline() for _ in range(n_verts)])), dtype=np.float32), \
                        np.asarray(list(map(
                    parse_single_face, [file.readline() for _ in range(n_faces)])), dtype=np.int32)

        def get_class(file_path: Path) -> str:
            """Returns the class of an OFF file belonging to the dataset."""
            return file_path.parents[1].name
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        points_paths = self._file_list[idx]
        if not type(points_paths) == list:
            points_paths = [points_paths]

        meshes = list(map(parse_off_points, points_paths))
        targets = list(map(get_class, points_paths))
        
        points = [mesh[0] for mesh in meshes]
        if self.sample:
            points = [self._sample_points_from_faces(mesh) for mesh in meshes]

        data = [(point, self.class_name_dict[target]) for point, target in zip(points, targets)]

        if self.transform is not None:
            data = [(self.transform(point), target) for point, target in data]
        if self.target_transform is not None:
            data = [(point, self.target_transform(target)) for point, target in data]

        if len(data) == 1:
            return data[0]
        return data

    def __len__(self) -> int:
        if self._file_list:
            return len(self._file_list)
        return 0

class Normalize:
    """Normalizes points in a point cloud to a sphere of unit radius."""
    def __call__(self, points: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        centered_points = points - np.mean(points, axis=0)
        normalized_points = centered_points / np.max(np.linalg.norm(centered_points, axis=1))
        return normalized_points

class RandomRotationZAxis:
    """Rotates a point cloud by a random angle theta about the z-axis"""

    def __call__(self, points: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        theta = random.random() * 2.0 * np.pi
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta),  np.cos(theta), 0],
                                    [           0,               0, 1]])
        return rotation_matrix.dot(points.T).T

class AddRandomNoise:
    """Adds Gaussian noise of specified mean and standard deviation to a point cloud."""
    def __init__(self, mean: float = 0.0, std_dev: float = 0.02) -> None:
        self._mean = mean
        self._std_dev = std_dev

    def __call__(self, points: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return points + np.random.normal(self._mean, self._std_dev, size=points.shape)

class PointToFloatTensor:
    """Returns a floating point tensor object for a given data point."""
    def __call__(self, points: int | float | np.ndarray[Any, Any]) -> torch.Tensor:
        return torch.tensor(points, dtype=torch.float)

class PointToLongTensor:
    """Returns a torch.long tensor object for a given data point."""
    def __call__(self, points: int | float | np.ndarray[Any, Any]) -> torch.Tensor:
        return torch.tensor(points, dtype=torch.long)

def train_val_split(data: torch.utils.data.Dataset, val_frac: float, seed: int = 42) -> \
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Splits a dataset into training and validation subsets."""
    num_val_samples = int(len(data) * val_frac)

    train_subset, val_subset = torch.utils.data.random_split(
            data, [len(data) - num_val_samples, num_val_samples], 
                   generator=torch.Generator().manual_seed(seed))
    return train_subset, val_subset

def load_training_and_validation_data(batch_size: int, val_frac: float = 0.2, augment: bool = True,
              num_workers: Optional[int] = None, seed: int = 42) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Loads training and validation data into dataloaders and returns the dataloaders"""
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
    data = ModelNet10(root=Path.cwd(), download=True, train=True, transform=data_transform, target_transform=PointToLongTensor())

    train, val = train_val_split(data, val_frac, seed)

    num_workers = os.cpu_count() - 1 if not num_workers else num_workers
 
    return DataLoader(train, batch_size=batch_size, num_workers=num_workers), \
    DataLoader(val, batch_size=batch_size, num_workers=num_workers)

def load_test_data(batch_size: int, augment: bool = True,
              num_workers: Optional[int] = None) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Loads training data into a dataloader and returns the dataloader"""
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
    data = ModelNet10(root=Path.cwd(), download=True, train=False, transform=data_transform, target_transform=PointToLongTensor())

    num_workers = os.cpu_count() - 1 if not num_workers else num_workers
 
    return DataLoader(data, batch_size=batch_size, num_workers=num_workers)
