import os
import glob
import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger
from .data import (
    load_zarr_store,
    setup_zarr_store,
    simulate_to_store,
    info,
    setup_dataloader,
)
from .config import get_settings
from .networks import setup_trainer, setup_logger
from .jsd import compute_doppel_factor


def simulate(simulator, settings):
    run_settings, data_settings, _, _, _ = get_settings(settings)
    if not run_settings["simulate"]:
        info(
            "Skipping simulations (run: simulate: False)",
            run_settings["verbose"],
        )
        return None
    if os.path.exists(path=data_settings["store_name"]):
        info(
            f"Loading ZarrStore: {data_settings['store_name']}",
            run_settings["verbose"],
        )
        store = load_zarr_store(data_settings["store_name"])
    else:
        info(
            f"Initialising ZarrStore: {data_settings['store_name']}",
            run_settings["verbose"],
        )
        store = setup_zarr_store(
            store_name=data_settings["store_name"],
            store_size=data_settings["store_size"],
            chunk_size=data_settings["chunk_size"],
            simulator=simulator,
        )
    if store.sims_required > 0:
        info(f"Simulating data into ZarrStore: {data_settings['store_name']}")
        simulate_to_store(
            store=store,
            simulator=simulator,
            batch_size=data_settings["chunk_size"],
            one_batch=False,
        )
        info(f"Simulations completed", run_settings["verbose"])
    else:
        info(
            f"ZarrStore ({data_settings['store_name']}) already full, skipping simulations",
            run_settings["verbose"],
        )


def train(network, settings):
    run_settings, data_settings, training_settings, _, _ = get_settings(
        settings
    )
    if not run_settings["train"]:
        info(
            "Skipping network training (run: train: False)",
            run_settings["verbose"],
        )
        return None
    info(
        f"Setting up dataloaders and trainer with:"
        + f"\ntrainer_dir\t | {training_settings['trainer_dir']}"
        + f"\ntrain_fraction\t | {training_settings['train_fraction']}"
        + f"\ntrain_batch_size | {training_settings['train_batch_size']}"
        + f"\nval_batch_size\t | {training_settings['val_batch_size']}"
        + f"\nnum_workers\t | {training_settings['num_workers']}",
        run_settings["verbose"],
    )

    if os.path.exists(path=data_settings["store_name"]):
        info(
            f"Loading ZarrStore: {data_settings['store_name']}",
            run_settings["verbose"],
        )
        store = load_zarr_store(data_settings["store_name"])
    else:
        raise ValueError(
            f"ZarrStore ({data_settings['store_name']}) does not exist, please simulate first"
        )
    if store.sims_required > 0:
        raise ValueError(
            f"ZarrStore ({data_settings['store_name']}) is not full, please simulate first"
        )

    train_data, val_data = setup_dataloader(
        store=store,
        num_workers=training_settings["num_workers"],
        train_fraction=training_settings["train_fraction"],
        train_batch_size=training_settings["train_batch_size"],
        val_batch_size=training_settings["val_batch_size"],
    )
    logger = setup_logger(training_settings)
    trainer = setup_trainer(
        device=training_settings["device"],
        n_devices=training_settings["n_devices"],
        min_epochs=training_settings["min_epochs"],
        max_epochs=training_settings["max_epochs"],
        logger=logger,
    )
    if not os.path.exists(training_settings["trainer_dir"]):
        os.mkdir(training_settings["trainer_dir"])
    trainer.fit(network, train_data, val_data)
    info(
        f"Training completed, checkpoint available at {glob.glob(training_settings['trainer_dir'] + '/doppel-epoch*.ckpt')[0]}",
        run_settings["verbose"],
    )


def sample(simulator, network, settings):
    (
        run_settings,
        _,
        training_settings,
        sampling_settings,
        _,
    ) = get_settings(settings)
    if not run_settings["sample"]:
        info(
            "Skipping sampling (run: sample: False)",
            run_settings["verbose"],
        )
        return None
    if (
        len(glob.glob(training_settings["trainer_dir"] + "/doppel-epoch*.ckpt"))
        == 0
    ):
        raise ValueError(
            f"No training checkpoint available in {training_settings['trainer_dir']}"
        )
    trainer = setup_trainer(
        device=training_settings["device"],
        n_devices=training_settings["n_devices"],
        min_epochs=training_settings["min_epochs"],
        max_epochs=training_settings["max_epochs"],
        logger=None,
    )
    initialiser_store = simulator.sample(training_settings["val_batch_size"])
    val_data = initialiser_store.get_dataloader(
        num_workers=0,
        batch_size=training_settings["val_batch_size"],
        on_after_load_sample=None,
    )
    trainer.test(
        network,
        val_data,
        glob.glob(training_settings["trainer_dir"] + "/doppel-epoch*.ckpt")[0],
    )

    sim_data = simulator.sample(
        sampling_settings.get("n_samples", 1000),
        targets=["data"],
        conditions={"model": np.array([0])},
    )
    real_data = simulator.sample(
        sampling_settings.get("n_samples", 1000),
        targets=["data"],
        conditions={"model": np.array([1])},
    )
    network.eval()
    with torch.no_grad():
        lp_sim_0 = np.array(
            network.forward(
                {"data": torch.tensor(sim_data["data"])},
                {"model": torch.tensor([[0]])},
            ).logratios
        ).T[0]
        lp_sim_1 = np.array(
            network.forward(
                {"data": torch.tensor(sim_data["data"])},
                {"model": torch.tensor([[1]])},
            ).logratios
        ).T[0]
        lp_real_0 = np.array(
            network.forward(
                {"data": torch.tensor(real_data["data"])},
                {"model": torch.tensor([[0]])},
            ).logratios
        ).T[0]
        lp_real_1 = np.array(
            network.forward(
                {"data": torch.tensor(real_data["data"])},
                {"model": torch.tensor([[1]])},
            ).logratios
        ).T[0]

    model_probabilities_saved_sims = np.zeros(
        (sampling_settings.get("n_samples", 1000), 3)
    )
    model_probabilities_saved_real = np.zeros(
        (sampling_settings.get("n_samples", 1000), 3)
    )

    model_probabilities_saved_sims[:, -2] = np.exp(lp_sim_0) / (
        np.exp(lp_sim_0) + np.exp(lp_sim_1)
    )
    model_probabilities_saved_sims[:, -1] = np.exp(lp_sim_1) / (
        np.exp(lp_sim_0) + np.exp(lp_sim_1)
    )
    model_probabilities_saved_sims[:, 0] = (
        model_probabilities_saved_sims[:, -1]
        / model_probabilities_saved_sims[:, -2]
    )

    model_probabilities_saved_real[:, -2] = np.exp(lp_real_0) / (
        np.exp(lp_real_0) + np.exp(lp_real_1)
    )
    model_probabilities_saved_real[:, -1] = np.exp(lp_real_1) / (
        np.exp(lp_real_0) + np.exp(lp_real_1)
    )
    model_probabilities_saved_real[:, 0] = (
        model_probabilities_saved_real[:, -1]
        / model_probabilities_saved_real[:, -2]
    )
    if not os.path.exists(sampling_settings["save_dir"]):
        os.mkdir(sampling_settings["save_dir"])
    np.save(
        f"{sampling_settings['save_dir']}" + f"/doppel-sample-sim-data.npy",
        np.array(model_probabilities_saved_sims),
    )
    np.save(
        f"{sampling_settings['save_dir']}" + f"/doppel-sample-real-data.npy",
        np.array(model_probabilities_saved_real),
    )
    info(
        f"Saved doppel sample results to {sampling_settings['save_dir']}",
        run_settings["verbose"],
    )


def doppel_factor(settings):
    (
        run_settings,
        _,
        _,
        sampling_settings,
        doppel_settings,
    ) = get_settings(settings)
    if not run_settings["doppel"]:
        info(
            "Skipping doppel factor calculation (run: doppel: False)",
            run_settings["verbose"],
        )
        return None
    if (
        len(
            glob.glob(
                sampling_settings["save_dir"] + "/doppel-sample-sim-data.npy"
            )
        )
        == 0
    ):
        raise ValueError(
            f"No sim data sampling results available in {sampling_settings['save_dir']}"
        )
    if (
        len(
            glob.glob(
                sampling_settings["save_dir"] + "/doppel-sample-real-data.npy"
            )
        )
        == 0
    ):
        raise ValueError(
            f"No real data sampling results available in {sampling_settings['save_dir']}"
        )
    sim_data_result = np.load(
        sampling_settings["save_dir"] + "/doppel-sample-sim-data.npy"
    )
    real_data_result = np.load(
        sampling_settings["save_dir"] + "/doppel-sample-real-data.npy"
    )
    doppel_factor = compute_doppel_factor(
        np.log10(sim_data_result[:, 0]),
        np.log10(real_data_result[:, 0]),
        nbins=doppel_settings["num_bins"],
        interpolation=doppel_settings["interpolation"],
        plot=doppel_settings["plot"],
        plot_dir=doppel_settings["plot_dir"],
    )
    info(
        f"Doppel factor: {doppel_factor}",
        run_settings["verbose"],
    )
    np.save(
        f"{sampling_settings['save_dir']}" + f"/doppel-factor.npy",
        np.array(doppel_factor),
    )
