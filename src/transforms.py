"""
Custom transform classes for point clouds to be used with torchvision.transforms.
Some of the transformation classes have been adapted from 
https://colab.research.google.com/drive/1K_RsM3db8bPrXsIcxV7Qv4cHJa-M2xSn.
"""
import random

from typing import Any

import torch
import numpy as np

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
