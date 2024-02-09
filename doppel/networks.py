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
        self.num_features = self.training_settings.get("num_features", 1)
        self.model_estimator = swyft.LogRatioEstimator_1dim(
            num_features=self.num_features, num_params=1, varnames="model"
        )

    def compression(self, data):
        return data

    def forward(self, A, B):
        data = A["data"]
        model = B["model"].float()
        summary = self.compression(data)
        return self.model_estimator(summary, model)

    def configure_callbacks(self):
        # TODO: [JA] Add learning rate monitor
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
        # TODO: [JA] Implement other schedulers
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=getattr(self, "lr_scheduler_factor", 0.1),
                patience=getattr(self, "lr_scheduler_patience", 3),
            ),
            "monitor": "val_loss",
        }
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler)


def setup_trainer(device, n_devices, min_epochs, max_epochs, logger=None):
    trainer = swyft.SwyftTrainer(
        accelerator=device,
        devices=n_devices,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        logger=logger,
    )
    return trainer


def setup_logger(settings):
    training_settings = settings.get("training", {"logger": {"type": None}})
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
            config=settings,
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


class ConvNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(ConvNet, self).__init__()
        self.num_channels = input_shape[0]
        self.hx, self.hy = input_shape[1], input_shape[2]

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                self.num_channels, 16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        conv_output_size = self._get_conv_output_size(input_shape)
        self.linear_layers = nn.Sequential(
            nn.Linear(conv_output_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    def _get_conv_output_size(self, shape):
        dummy_input = torch.zeros(1, *shape)
        output = self.conv_layers(dummy_input)
        output_size = output.view(output.size(0), -1).size(1)
        return output_size


class LinearCompression(nn.Module):
    def __init__(self, hidden_size=10, n_features=1):
        super(LinearCompression, self).__init__()
        self.sequential = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
            nn.LazyLinear(n_features),
        )

    def forward(self, x):
        return self.sequential(x)
