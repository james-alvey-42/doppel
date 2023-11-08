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

from doppel import Simulator, RatioEstimator, setup_trainer, load_npz
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

dataset_library = {
    "cifar": "data/cifar_test_samples_10000x32x32x3.npz",
    "EDM_ES": "data/EDM_ES_test_samples_10000x32x32x3.npz",
    "pfgmpp": "data/pfgmpp_test_samples_10000x32x32x3.npz",
    "GAN": "data/GAN_test_samples_10000x32x32x3.npz"
}


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    dataset1 = args[0]
    dataset2 = args[1]
    diff_data = load_npz(dataset_library[dataset1])
    print(diff_data.shape)
    cifar_data = load_npz(dataset_library[dataset2])
    print(cifar_data.shape)

    diff_data = diff_data[:5000]
    cifar_data = cifar_data[5000:]

    config = {
        "store_name": f"{dataset1}-{dataset2}-store",
        "store_size": 40_000,
        "chunk_size": 500,
        "observation_path": None,
        "logratios_path": f"{dataset1}-{dataset2}-logratios",
        "trainer_dir": f"{dataset1}-{dataset2}-trainer",
        "resampler_targets": ["data"],
        "train_fraction": 0.8,
        "train_batch_size": 256,
        "val_batch_size": 256,
        "num_workers": 8,
        "device": "gpu",
        "n_gpus": 1,
        "min_epochs": 1,
        "max_epochs": 100,
        "early_stopping": 15,
        "infer_only": False,
    }
    simulator = Simulator(diff_data, cifar_data)
    trainer = setup_trainer(
        trainer_dir=config["trainer_dir"],
        early_stopping=config["early_stopping"],
        device=config["device"],
        n_gpus=config["n_gpus"],
        min_epochs=config["min_epochs"],
        max_epochs=config["max_epochs"],
    )
    network = RatioEstimator(input_shape=simulator.sample()["data"].shape)
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

    model_probabilities_saved_sims = np.zeros((5000, 3))
    for i in range(0, 5000):
        sim_observation = simulator.sample(
            targets=["data"], conditions={"model": np.array([0])}
        )
        logratios = trainer.infer(
            network, sim_observation, prior_samples.get_dataloader(batch_size=2048)
        )
        model_probabilities = get_class_probs(logratios, params=["model[0]"])
        C0 = model_probabilities[1] / model_probabilities[0]
        model_probabilities_saved_sims[i, -2:] = model_probabilities
        model_probabilities_saved_sims[i, 0] = C0
    np.save(
        f"result-sims-{dataset1}-{dataset2}.npy",
        np.array(model_probabilities_saved_sims),
    )

    model_probabilities_saved_real = np.zeros((5000, 3))
    for i in range(0, 5000):
        sim_observation = simulator.sample(
            targets=["data"], conditions={"model": np.array([1])}
        )
        logratios = trainer.infer(
            network, sim_observation, prior_samples.get_dataloader(batch_size=2048)
        )
        model_probabilities = get_class_probs(logratios, params=["model[0]"])
        C0 = model_probabilities[1] / model_probabilities[0]
        model_probabilities_saved_real[i, -2:] = model_probabilities
        model_probabilities_saved_real[i, 0] = C0
    np.save(
        f"result-real-{dataset1}-{dataset2}.npy",
        np.array(model_probabilities_saved_real),
    )

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.hist(
        np.log10(model_probabilities_saved_sims[:, 0]),
        bins=20,
        alpha=0.5,
        color="tab:blue",
    )
    ax.hist(
        np.log10(model_probabilities_saved_real[:, 0]),
        bins=20,
        alpha=0.5,
        color="tab:red",
    )
    plt.savefig(f'viewer-{dataset1}-{dataset2}.png')
