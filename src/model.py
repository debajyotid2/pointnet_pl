"""Point net model implemented in PyTorch and PyTorch Lightning"""

import torch
import pytorch_lightning as pl
from torch.nn import (
            Module, Linear, ReLU, 
            Sequential, BatchNorm1d, MaxPool1d,
            Dropout, Conv1d, Flatten
        )
from typing import Callable


class Transpose(Module):
    """
    Transposes the last two dimensions of a tensor, i.e. from (N, L, C) to (N, C, L). Takes as an 
    argument the dim1 of a tensor of shape (dim0, dim1, dim2) that is expected by the next layer in sequence.
    Preferably used with Sequential modules."""
    def __init__(self, dim1: int) -> None:
        super().__init__()
        self.dim1 = dim1

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.size(-2) != self.dim1:
            return input_tensor.transpose(1, 2)
        return input_tensor

class FCBlock(Module):
    """Building block of fully connected layers for pointnet network"""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.block = Sequential(
                    Linear(in_features, out_features),
                    BatchNorm1d(out_features),
                    ReLU()
                )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.block(input_tensor)

class ConvBlock(Module):
    """Building block of convolution layers for pointnet network"""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.block = Sequential(
                    Conv1d(in_features, out_features, kernel_size=1),
                    BatchNorm1d(out_features),
                    ReLU()
                )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.block(input_tensor)

class TransformationNetwork(Module):
    """Network defining the transformation h(.) applied to points before feature extraction"""
    def __init__(self, input_shape: tuple[int], in_features: int, out_features: int) -> None:
        super().__init__()
        self.t_net = Sequential(
                    Transpose(in_features),
                    ConvBlock(in_features, 64),
                    ConvBlock(64, 128),
                    ConvBlock(128, 1024),
                    MaxPool1d(input_shape[0]),
                    Flatten(),
                    FCBlock(1024, 512),
                    FCBlock(512, 256),
                    Linear(256, out_features * out_features),
                )
        self.out_features = out_features

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t_net_out = self.t_net(input_tensor).view(input_tensor.size(0), self.out_features, self.out_features) 
        t_net_out += torch.eye(self.out_features, requires_grad=True, device=input_tensor.device).repeat(input_tensor.size(0), 1, 1)
        if input_tensor.size(-2) != self.out_features:
            reshaped_input = input_tensor.transpose(1, 2)
        else:
            reshaped_input = input_tensor
        output_tensor = torch.matmul(t_net_out, reshaped_input)
        return output_tensor, t_net_out

class PointNetClassifier(pl.LightningModule):
    """Pytorch Lightning Module for PointNet classifier training."""
    def __init__(self,
                 input_transform: torch.nn.Module,
                 feature_transform: torch.nn.Module,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 input_shape: tuple[int],
                 num_classes: int,
                 feature_mlp_out_ftrs: list[int],
                 global_feature_mlp_out_ftrs: list[int],
                 learning_rate: float = 1e-3,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 regularization_weight: float = 0.0001,
                 scheduler_stepsize: int = 20,
                 scheduler_gamma: float = 0.5,
                 dropout_p: float = 0.7) -> None:
        super().__init__()
        print(self.hparams)
        self.save_hyperparameters(ignore=["loss_fn", "input_transform", "feature_transform"])

        self.loss_fn = loss_fn
        
        self.input_transform = input_transform
        self.input_feature_mlp = Sequential(
                    Transpose(3),
                    ConvBlock(3, 64),
                    ConvBlock(64, 64)
                )
        self.feature_transform = feature_transform
        self.feature_mlp = Sequential(
                    Transpose(64),
                    ConvBlock(64, feature_mlp_out_ftrs[0]),
                    ConvBlock(feature_mlp_out_ftrs[0], feature_mlp_out_ftrs[1]),
                    ConvBlock(feature_mlp_out_ftrs[1], feature_mlp_out_ftrs[2]),
                )
        self.maxpool = MaxPool1d(input_shape[0])
        self.flatten = Flatten()
        self.global_feature_mlp = Sequential(
                    FCBlock(feature_mlp_out_ftrs[-1], global_feature_mlp_out_ftrs[0]),
                    Dropout(dropout_p),
                    FCBlock(global_feature_mlp_out_ftrs[0], global_feature_mlp_out_ftrs[1]),
                    Dropout(dropout_p),
                    Linear(global_feature_mlp_out_ftrs[1], num_classes),
                )

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transformed_input, _ = self.input_transform(input_tensor)
        feature_tensor = self.input_feature_mlp(transformed_input)
        transformed_features, t_net_out_64by64 = self.feature_transform(feature_tensor)
        global_features_tensor = self.maxpool(self.feature_mlp(transformed_features))
        return self.global_feature_mlp(self.flatten(global_features_tensor)), t_net_out_64by64
                

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.tensor:
        points, targets = batch
        preds, t_net_out_64by64 = self.forward(points)
        loss = self.loss_fn(preds, targets)
        loss += self.regularization_term(t_net_out_64by64)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        points, targets = batch
        preds, t_net_out_64by64 = self.forward(points)
        loss = self.loss_fn(preds, targets)
        loss += self.regularization_term(t_net_out_64by64)
        accuracy = self.get_accuracy(preds, targets)
        self.log_dict(dict(val_loss = loss, val_acc = accuracy))
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        points, targets = batch
        preds, t_net_out_64by64 = self.forward(points)
        accuracy = self.get_accuracy(preds, targets)
        self.log("test_acc", accuracy)
        return accuracy

    def get_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculates accuracy given logits and targets."""
        preds = torch.argmax(logits.softmax(dim=1), dim=1)
        correct = torch.eq(preds, targets)
        accuracy = torch.sum(correct.cpu().apply_(lambda x: 1.0 if x else 0.0), dim=0) / targets.size(0)
        return accuracy

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(params = self.parameters(),
                                    lr=self.hparams.learning_rate,
                                    betas=(self.hparams.beta1, self.hparams.beta2)
                                )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=self.hparams.scheduler_stepsize, 
                                                    gamma=self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]

    def regularization_term(self, t_net_out_64by64: torch.Tensor) -> torch.Tensor:
        """Returns the regularization term to be added to the softmax loss."""
        def calc_regularization_term(t_net_out: torch.Tensor) -> torch.Tensor:
            identity_mat = (torch.eye(n=t_net_out.size(1), device=t_net_out.device)
                            .view(1, t_net_out.size(1), t_net_out.size(1))
                            .repeat(t_net_out.size(0), 1, 1)) 
            diff = identity_mat - torch.matmul(t_net_out, torch.transpose(t_net_out, dim0=1, dim1=2))
            return torch.mean(torch.square(torch.linalg.matrix_norm(diff)))

        return self.hparams.regularization_weight * calc_regularization_term(t_net_out_64by64)

class PointNetClassifierNoTransforms(pl.LightningModule):
    """Pytorch Lightning Module for PointNet classifier training without transforms."""
    def __init__(self,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 input_shape: tuple[int],
                 num_classes: int,
                 feature_mlp_out_ftrs: list[int],
                 global_feature_mlp_out_ftrs: list[int],
                 learning_rate: float = 1e-3,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 regularization_weight: float = 0.0001,
                 scheduler_stepsize: int = 20,
                 scheduler_gamma: float = 0.5,
                 dropout_p: float = 0.7) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["loss_fn"])

        self.loss_fn = loss_fn
        
        self.input_feature_mlp = Sequential(
                    Transpose(3),
                    ConvBlock(3, 64),
                    ConvBlock(64, 64)
                )
        self.feature_mlp = Sequential(
                    Transpose(64),
                    ConvBlock(64, feature_mlp_out_ftrs[0]),
                    ConvBlock(feature_mlp_out_ftrs[0], feature_mlp_out_ftrs[1]),
                    ConvBlock(feature_mlp_out_ftrs[1], feature_mlp_out_ftrs[2]),
                )
        self.maxpool = MaxPool1d(input_shape[0])
        self.flatten = Flatten()
        self.global_feature_mlp = Sequential(
                    FCBlock(feature_mlp_out_ftrs[-1], global_feature_mlp_out_ftrs[0]),
                    Dropout(dropout_p),
                    FCBlock(global_feature_mlp_out_ftrs[0], global_feature_mlp_out_ftrs[1]),
                    Dropout(dropout_p),
                    Linear(global_feature_mlp_out_ftrs[1], num_classes),
                )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        feature_tensor = self.input_feature_mlp(input_tensor)
        global_features_tensor = self.maxpool(self.feature_mlp(feature_tensor))
        return self.global_feature_mlp(self.flatten(global_features_tensor))

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.tensor:
        points, targets = batch
        preds = self.forward(points)
        loss = self.loss_fn(preds, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        points, targets = batch
        preds = self.forward(points)
        loss = self.loss_fn(preds, targets)
        accuracy = self.get_accuracy(preds, targets)
        self.log_dict(dict(val_loss = loss, val_acc = accuracy))
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        points, targets = batch
        preds = self.forward(points)
        accuracy = self.get_accuracy(preds, targets)
        self.log("test_acc", accuracy)
        return accuracy

    def get_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculates accuracy given logits and targets."""
        preds = torch.argmax(logits.softmax(dim=1), dim=1)
        correct = torch.eq(preds, targets)
        accuracy = torch.sum(correct.cpu().apply_(lambda x: 1.0 if x else 0.0), dim=0) / targets.size(0)
        return accuracy

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(params = self.parameters(),
                                    lr=self.hparams.learning_rate,
                                    betas=(self.hparams.beta1, self.hparams.beta2)
                                )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.scheduler_stepsize, gamma=self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]


