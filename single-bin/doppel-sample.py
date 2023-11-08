if __name__ == "__main__":
    print(
        """
  ____    ____    ____    ____   _____   _      
 |  _ \  / __ \  |  _ \  |  _ \ | ____| | |        DOPPEL | Assessing model quality with ML  
 | | | || |  | | | |_) | | |_) || |_    | |        Model: Single Bin
 | | | || |  | | |  __/  |  __/ | __|   | |        Module: doppel-sample.py
 | |/ / | |__| | | |     | |    | |___  | |___     Authors: J. Alvey, T. Edwards
 |___/   \____/  |_|     |_|    |_____| |_____|    Version: 0.1
    """
    )

from doppel import Simulator, RatioEstimator, setup_trainer
import glob
import numpy as np
import swyft


def get_class_probs(lrs, params):
    params, weights = swyft.get_weighted_samples(lrs, params)
    probs = np.array(
        [weights[params[:, 0] == k].sum() for k in range(int(params[:, 0].max()) + 1)]
    )
    probs /= probs.sum()
    return probs


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    bkg = int(args[0])
    amp = int(args[1])
    config = {
        "store_name": f"single-bin-store-{bkg}-{amp}",
        "store_size": 100_000,
        "chunk_size": 500,
        "observation_path": None,
        "logratios_path": f"single-bin-logratios-{bkg}-{amp}",
        "trainer_dir": f"single-bin-trainer-{bkg}-{amp}",
        "resampler_targets": ["data"],
        "train_fraction": 0.85,
        "train_batch_size": 512,
        "val_batch_size": 512,
        "num_workers": 8,
        "device": "gpu",
        "n_gpus": 1,
        "min_epochs": 1,
        "max_epochs": 80,
        "early_stopping": 15,
        "infer_only": False,
    }
    simulator = Simulator(bkg=bkg, amp=amp)
    trainer = setup_trainer(
        trainer_dir=config["trainer_dir"],
        early_stopping=config["early_stopping"],
        device=config["device"],
        n_gpus=config["n_gpus"],
        min_epochs=config["min_epochs"],
        max_epochs=config["max_epochs"],
    )
    network = RatioEstimator()
    val_data_store = simulator.sample(config["val_batch_size"])
    val_data = val_data_store.get_dataloader(
        num_workers=config["num_workers"],
        batch_size=config["val_batch_size"],
        on_after_load_sample=None,
    )
    trainer.test(
        network, val_data, glob.glob(config["trainer_dir"] + "/epoch*.ckpt")[0]
    )
    prior_samples = swyft.Samples({"model": np.array([[0], [1]])})

    model_probabilities_saved_sims = np.zeros((10000, 4))
    for i in range(0, 10000):
        sim_observation = simulator.sample(
            targets=["data"], conditions={"model": np.array([0])}
        )
        logratios = trainer.infer(
            network, sim_observation, prior_samples.get_dataloader(batch_size=2048)
        )
        model_probabilities = get_class_probs(logratios, params=["model[0]"])
        num_events = bkg * sim_observation["data"]
        C0 = model_probabilities[1] / model_probabilities[0]
        model_probabilities_saved_sims[i, -2:] = model_probabilities
        model_probabilities_saved_sims[i, 0] = num_events
        model_probabilities_saved_sims[i, 1] = C0
    np.save(
        f"result-{bkg}-{amp}.npy",
        np.array(model_probabilities_saved_sims),
    )
