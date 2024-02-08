import torch
import torch.nn as nn
import swyft
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

if swyft.__version__ != "0.4.5":
    raise ImportError("ERROR: swyft version must be 0.4.5")


class DoppelRatioEstimator(swyft.SwyftModule, swyft.AdamWReduceLROnPlateau):
    def __init__(self, settings={}):
        super().__init__()
        self.compression_settings = settings.get(
            "compression", {"use_compression": False, "num_features": 1}
        )
        self.training_settings = settings.get("training", {})
        self.training_dir = self.training_settings.get(
            "training_dir", "doppel_training"
        )
        self.learning_rate = self.training_settings.get("learning_rate", 1e-4)
        self.early_stopping_patience = self.training_settings.get(
            "early_stopping_patience", 7
        )
        self.lr_scheduler_factor = self.training_settings.get(
            "lr_scheduler_factor", 0.1
        )
        self.lr_scheduler_patience = self.training_settings.get(
            "lr_scheduler_patience", 3
        )

        if not self.compression_settings["use_compression"]:
            self.compression = None
        else:
            self.compression = self.setup_compression()
        self.num_features = self.compression_settings["num_features"]
        self.model_estimator = swyft.LogRatioEstimator_1dim(
            num_features=self.num_features, num_params=1, varnames="model"
        )

    def forward(self, A, B):
        data = A["data"]
        model = B["model"].float()
        if self.compression is not None:
            summary = self.compression(data)
            return self.model_estimator(summary, model)
        else:
            return self.model_estimator(data, model)

    def setup_compression(self):
        # TODO: [JA] Implement compressions
        self.compression = None

    def configure_callbacks(self):
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=getattr(self, "early_stopping_patience", 7),
        )
        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=getattr(self, "training_dir", "doppel_training"),
            filename="doppel-{epoch:02d}-{val_loss:.3f}",
        )
        return [early_stop, checkpoint]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=getattr(self, "learning_rate", 1e-4)
        )
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=getattr(self, "lr_scheduler_factor", 0.1),
                patience=getattr(self, "lr_scheduler_patience", 3),
            ),
            "monitor": "val_loss",
        }
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler)


class LinearCompression(nn.Module):
    def __init__(self, hidden_size=16, n_features=1):
        super(LinearCompression, self).__init__()
        self.sequential = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
            nn.LazyLinear(n_features),
        )

    def forward(self, x):
        return self.sequential(x)


def setup_trainer(device, n_devices, min_epochs, max_epochs, logger=None):
    trainer = swyft.SwyftTrainer(
        accelerator=device,
        devices=n_devices,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        logger=logger,
    )
    return trainer


def setup_logger(training_settings):
    if training_settings["logger"]["type"] is None:
        logger = None
    elif training_settings["logger"]["type"] == "wandb":
        if "entity" not in training_settings["logger"].keys():
            raise ValueError("(entity) is a required field for WandB logger")
        logger = WandbLogger(
            offline=training_settings["logger"].get("offline", False),
            name=training_settings["logger"].get("name", "doppel_run"),
            project=training_settings["logger"].get("project", "doppel"),
            entity=training_settings["logger"]["entity"],
            log_model=training_settings["logger"].get("log_model", "all"),
        )
    elif training_settings["logger"]["type"] == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=training_settings["logger"].get(
                "save_dir", "doppel_training"
            ),
            name=training_settings["logger"].get("name", "doppel_run"),
            version=None,
            default_hp_metric=False,
        )
    else:
        logger = None
        print(
            f"Logger: {training_settings['logger']['type']} not implemented, logging disabled"
        )
    return logger
