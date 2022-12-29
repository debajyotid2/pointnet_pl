import random

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from src.model import TransformationNetwork, PointNetClassifier, PointNetClassifierNoTransforms
from src.dataset import load_training_and_validation_data, load_test_data

DATASET = "ModelNet40"
NUM_CLASSES = 40
FEATURE_MLP_OUT_FTRS = [64, 128, 1024]
GLOBAL_FEATURE_MLP_OUT_FTRS = [64, 128, 1024]
LEARNING_RATE = 1e-3
BETA1 = 0.9
BETA2 = 0.999
DROPOUT_P = 0.7
BATCH_SIZE = 32
TEST_BATCH_SIZE = 128
MAX_EPOCHS = 1
TRANSFORMS = True
SEED = 42
AUGMENT = True

PL_WORKING_DIR = Path(".").resolve()
CKPT_PATH = None

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def main() -> None:
    train_ds, val_ds = load_training_and_validation_data(dataset=DATASET, batch_size=BATCH_SIZE, augment=AUGMENT)
    test_ds = load_test_data(dataset=DATASET, batch_size=TEST_BATCH_SIZE, augment=AUGMENT)
        
    ckpt_path = CKPT_PATH
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path) 

    transforms = TRANSFORMS

    input_shape = next(iter(train_ds))[0].shape[1:]
    
    loss_fn = torch.nn.functional.cross_entropy
    
    if transforms:
        classifier = PointNetClassifier(
                    input_transform=TransformationNetwork(input_shape, 3, 3),
                    feature_transform=TransformationNetwork(input_shape, 64, 64),
                    loss_fn=loss_fn,
                    input_shape=input_shape,
                    num_classes=NUM_CLASSES,
                    feature_mlp_out_ftrs=FEATURE_MLP_OUT_FTRS,
                    global_feature_mlp_out_ftrs=GLOBAL_FEATURE_MLP_OUT_FTRS,
                    learning_rate=LEARNING_RATE,
                    beta1=BETA1,
                    beta2=BETA2,
                    dropout_p=DROPOUT_P
                )
    else:
        classifier = PointNetClassifierNoTransforms(
                    loss_fn=loss_fn,
                    input_shape=input_shape,
                    num_classes=NUM_CLASSES,
                    feature_mlp_out_ftrs=FEATURE_MLP_OUT_FTRS,
                    global_feature_mlp_out_ftrs=GLOBAL_FEATURE_MLP_OUT_FTRS,
                    learning_rate=LEARNING_RATE,
                    beta1=BETA1,
                    beta2=BETA2,
                    dropout_p=DROPOUT_P
                )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer(default_root_dir=PL_WORKING_DIR,
                         accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1,
                         max_epochs=MAX_EPOCHS,
                         callbacks=[checkpoint_callback])
    if ckpt_path is not None:
        trainer.fit(classifier, 
                train_dataloaders=train_ds, 
                val_dataloaders=val_ds, 
                ckpt_path=ckpt_path
            )
    else:
        trainer.fit(classifier,
                    train_dataloaders=train_ds,
                    val_dataloaders=val_ds,
                )
    trainer.test(classifier, dataloaders=test_ds)

if __name__ == "__main__":
    main()
