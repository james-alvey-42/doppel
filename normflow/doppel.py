if __name__ == "__main__":
    print(
        """
  ____    ____    ____    ____   _____   _      
 |  _ \  / __ \  |  _ \  |  _ \ | ____| | |        DOPPEL | Assessing model quality with ML  
 | | | || |  | | | |_) | | |_) || |_    | |        Model: Normalising Flow
 | | | || |  | | |  __/  |  __/ | __|   | |        Module: doppel.py
 | |/ / | |__| | | |     | |    | |___  | |___     Authors: J. Alvey, T. Edwards
 |___/   \____/  |_|     |_|    |_____| |_____|    Version: 0.1
    """
    )

from typing import Sequence, Union
from pytorch_lightning.callbacks.callback import Callback
import swyft
import swyft.lightning as sl
import numpy as np
import os
import glob
import torch
import torch.nn as nn
from datetime import datetime
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

"""
--------------------------------------------------------------
SIMULATOR AND CLASS ESTIMATOR NETWORKS (REQUIRES MODIFICATION)
--------------------------------------------------------------
"""


class Simulator(swyft.Simulator):
    def __init__(self, nfm_data, target_data):
        super().__init__()
        self.nfm_data = nfm_data
        self.target_data = target_data
        self.transform_samples = swyft.to_numpy32

    def sample_model(self):
        return np.array([np.random.choice(2)])

    def generate_sim_data(self):
        return self.nfm_data[np.random.choice(self.nfm_data.shape[0]), :]

    def generate_real_data(self):
        return self.target_data[np.random.choice(self.target_data.shape[0]), :]

    def sample_data(self, model):
        if model == 0:
            return self.generate_sim_data()
        elif model == 1:
            return self.generate_real_data()
        else:
            err_msg = f"ERROR: model must be in [0, 1] | model = {model}"
            raise ValueError(err_msg)

    def build(self, graph):
        model = graph.node("model", self.sample_model)
        data = graph.node("data", self.sample_data, model)


class RatioEstimator(swyft.SwyftModule):
    def __init__(self):
        super().__init__()
        self.compression = None
        self.model_estimator = swyft.LogRatioEstimator_1dim(
            num_features=2, num_params=1, varnames="model"
        )
        self.online_normalisation = swyft.networks.OnlineStandardizingLayer(
            shape=torch.Size([2])
        )
        self.learning_rate = 1e-4
        self.early_stopping_patience = 7
        #self.optimizer_init = swyft.AdamOptimizerInit(lr=1e-4)

    def configure_callbacks(self):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=7,
            verbose=False,
            mode="min",
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=f"my_path",
            filename="{epoch}_{val_loss:.2f}_{train_loss:.2f}",
            mode="min",
        )
        return [lr_monitor, early_stopping_callback, checkpoint_callback]

    def forward(self, A, B):
        data = A["data"]
        model = B["model"].float()

        if self.compression is not None:
            data = self.online_normalisation(data)
            summary = self.compression(data)
            return self.model_estimator(summary, model)
        else:
            return self.model_estimator(data, model)


"""
--------------------------------------------
UTILITY FUNCTIONS (NO MODIFICATION REQUIRED)
--------------------------------------------
"""


def setup_zarr_store(store_name, store_size, chunk_size, simulator):
    store = swyft.ZarrStore(file_path=store_name)
    shapes, dtypes = simulator.get_shapes_and_dtypes()
    store.init(N=store_size, chunk_size=chunk_size, shapes=shapes, dtypes=dtypes)
    return store


def load_zarr_store(store_path):
    if os.path.exists(path=store_path):
        return swyft.ZarrStore(file_path=store_path)
    else:
        raise ValueError(f"store path ({store_path}) does not exist")


def simulate(store, simulator, batch_size, one_batch=False):
    if one_batch:
        store.simulate(sampler=simulator, batch_size=batch_size, max_sims=batch_size)
    else:
        store.simulate(sampler=simulator, batch_size=batch_size)


def info(msg):
    print(f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [real-data.py] | {msg}")


def config_info(config):
    print("--------\nSETTINGS\n--------")
    for key in config.keys():
        print(key + " | " + str(config[key]))
    print("\n")


def setup_dataloader(
    store,
    simulator,
    trainer_dir,
    num_workers,
    train_fraction,
    train_batch_size,
    val_batch_size,
    resampler_targets=None,
):
    if not os.path.isdir(trainer_dir):
        os.mkdir(trainer_dir)
    if resampler_targets is not None:
        if type(resampler_targets) is not list:
            raise ValueError(
                f"resampler targets must be a list | resampler_targets = {resampler_targets}"
            )
        resampler = simulator.get_resampler(targets=resampler_targets)
    else:
        resampler = None
    train_data = store.get_dataloader(
        num_workers=num_workers,
        batch_size=train_batch_size,
        idx_range=[0, int(train_fraction * len(store))],
        on_after_load_sample=resampler,
    )
    val_data = store.get_dataloader(
        num_workers=num_workers,
        batch_size=val_batch_size,
        idx_range=[
            int(train_fraction * len(store)),
            len(store) - 1,
        ],
        on_after_load_sample=None,
    )
    return train_data, val_data


def setup_trainer(trainer_dir, early_stopping, device, n_gpus, min_epochs, max_epochs):
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=early_stopping,
        verbose=False,
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{trainer_dir}",
        filename="{epoch}_{val_loss:.2f}_{train_loss:.2f}",
        mode="min",
    )
    logger_tbl = pl_loggers.TensorBoardLogger(
        save_dir=trainer_dir,
        name="swyft-trainer",
        version=None,
        default_hp_metric=False,
    )
    trainer = sl.SwyftTrainer(
        accelerator=device,
        gpus=n_gpus,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        logger=logger_tbl,
        callbacks=[lr_monitor, 
                   early_stopping_callback, 
                   checkpoint_callback],
    )
    return trainer


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    epoch = args[0]
    nfm_data = np.load(f"data/nfm_sample_{epoch}.npy")
    print(nfm_data)
    target_data = np.load("data/target.npy")
    print(target_data)

    config = {
        "store_name": f"nfm-store-{epoch}",
        "store_size": 100_000,
        "chunk_size": 500,
        "observation_path": None,
        "logratios_path": f"nfm-logratios-{epoch}",
        "trainer_dir": f"nfm-trainer-{epoch}",
        "resampler_targets": ["data"],
        "train_fraction": 0.9,
        "train_batch_size": 512,
        "val_batch_size": 512,
        "num_workers": 8,
        "device": "gpu",
        "n_gpus": 1,
        "min_epochs": 1,
        "max_epochs": 100,
        "early_stopping": 7,
        "infer_only": False,
    }
    config_info(config=config)
    simulator = Simulator(nfm_data, target_data)

    if os.path.exists(path=config["store_name"]):
        info(f"Loading ZarrStore: {config['store_name']}")
        store = load_zarr_store(config["store_name"])
    else:
        info(f"Initialising ZarrStore: {config['store_name']}")
        store = setup_zarr_store(
            store_name=config["store_name"],
            store_size=config["store_size"],
            chunk_size=config["chunk_size"],
            simulator=simulator,
        )

    print(
        "\n----------------------------\nSTEP 1: GENERATE SIMULATIONS\n----------------------------\n"
    )
    if store.sims_required > 0:
        info(f"Simulating data into ZarrStore: {config['store_name']}")
        simulate(
            store=store,
            simulator=simulator,
            batch_size=config["chunk_size"],
            one_batch=False,
        )
        info(f"Simulations completed")
    else:
        info(f"ZarrStore ({config['store_name']}) already full, skipping simulations")
    print(
        "\n-----------------------\nSTEP 2: TRAIN ESTIMATOR\n-----------------------\n"
    )
    info(
        f"Setting up dataloaders and trainer with:"
        + f"\ntrainer_dir\t | {config['trainer_dir']}"
        + f"\ntrain_fraction\t | {config['train_fraction']}"
        + f"\ntrain_batch_size | {config['train_batch_size']}"
        + f"\nval_batch_size\t | {config['val_batch_size']}"
        + f"\nnum_workers\t | {config['num_workers']}"
    )
    train_data, val_data = setup_dataloader(
        store=store,
        simulator=simulator,
        trainer_dir=config["trainer_dir"],
        num_workers=config["num_workers"],
        train_fraction=config["train_fraction"],
        train_batch_size=config["train_batch_size"],
        val_batch_size=config["val_batch_size"],
        resampler_targets=config["resampler_targets"],
    )
    trainer = setup_trainer(
        trainer_dir=config["trainer_dir"],
        early_stopping=config["early_stopping"],
        device=config["device"],
        n_gpus=config["n_gpus"],
        min_epochs=config["min_epochs"],
        max_epochs=config["max_epochs"],
    )
    network = RatioEstimator()
    if (
        not config["infer_only"]
        or len(glob.glob(f"{config['trainer_dir']}/epoch*.ckpt")) == 0
    ):
        info(f"Starting ratio estimator training")
        trainer.fit(network, train_data, val_data)
        info(
            f"Training completed, checkpoint available at {glob.glob(config['trainer_dir'] + '/epoch*.ckpt')[0]}"
        )
        info("To avoid re-training network, set infer_only = True in config")
    info(
        f"Loading optimal network weights from {glob.glob(config['trainer_dir'] + '/epoch*.ckpt')[0]}"
    )
    trainer.test(
        network, val_data, glob.glob(config["trainer_dir"] + "/epoch*.ckpt")[0]
    )
