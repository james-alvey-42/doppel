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
import torch
import tqdm


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
    epoch = args[0]
    nfm_data = np.load(f"data/nfm_sample_{epoch}.npy")
    print(nfm_data)
    target_data = np.load("target_test.npy")
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
    simulator = Simulator(nfm_data, target_data)
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

    Nsim = 1000
    model_probabilities_saved_sims = np.zeros((Nsim, 3))
    # sim_observations = simulator.sample(Nsim,
    #         targets=["data"], conditions={"model": np.array([0])}
    #     )
    # with torch.no_grad():
    #     lp_sim_0 = network({'data': torch.tensor(sim_observations['data'])}, {'model': torch.tensor(sim_observations['model'])}).logratios
    #     lp_sim_1 = network({'data': torch.tensor(sim_observations['data'])}, {'model': torch.tensor([[1]])}).logratios
    # print(lp_sim_0)
    # print(lp_sim_1)
    network.eval()
    for i in tqdm.tqdm(range(0, Nsim)):
        sim_observation = simulator.sample(
            targets=["data"], conditions={"model": np.array([0])}
        )
        #sim_observation = sim_observations[i]
        # logratios = trainer.infer(
        #     network, sim_observation, prior_samples.get_dataloader(batch_size=2048)
        # )
        # print(get_class_probs(logratios, params=["model[0]"]))
        with torch.no_grad():
            lp_sim = network.forward({'data': torch.tensor([sim_observation['data']])}, {'model': torch.tensor([[0], [1]])})
        # # print(lp_sim.logratios)
        # model_probabilities = get_class_probs(logratios, params=["model[0]"])
        # print(model_probabilities)

        model_probabilities2 = get_class_probs(lp_sim, params=["model[0]"])
        print(model_probabilities2)

        C0 = model_probabilities2[1] / model_probabilities2[0]
        model_probabilities_saved_sims[i, -2:] = model_probabilities2
        model_probabilities_saved_sims[i, 0] = C0
    np.save(
        f"result-sims-{epoch}.npy",
        np.array(model_probabilities_saved_sims),
    )

    model_probabilities_saved_real = np.zeros((Nsim, 3))
    # sim_observations = simulator.sample(Nsim,
    #         targets=["data"], conditions={"model": np.array([1])}
    #     )
    # with torch.no_grad():
    #     lp_sim_0 = network({'data': torch.tensor(sim_observations['data'])}, {'model': torch.tensor(sim_observations['model'])}).logratios
    #     lp_sim_1 = network({'data': torch.tensor(sim_observations['data'])}, {'model': torch.tensor([[1]])}).logratios
    # print(lp_sim_0)
    # print(lp_sim_1)
    for i in tqdm.tqdm(range(0, Nsim)):
        real_observation = simulator.sample(
            targets=["data"], conditions={"model": np.array([1])}
        )
        #sim_observation = sim_observations[i]
        # logratios = trainer.infer(
        #     network, real_observation, prior_samples.get_dataloader(batch_size=2048)
        # )
        with torch.no_grad():
            lp_real = network.forward({'data': torch.tensor([real_observation['data']])}, {'model': torch.tensor([[0], [1]])})

        # model_probabilities = get_class_probs(logratios, params=["model[0]"])
        # print(model_probabilities)

        model_probabilities2 = get_class_probs(lp_real, params=["model[0]"])
        print(model_probabilities2)

        C0 = model_probabilities2[1] / model_probabilities2[0]
        model_probabilities_saved_real[i, -2:] = model_probabilities2
        model_probabilities_saved_real[i, 0] = C0
    np.save(
        f"result-real-{epoch}.npy",
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
    plt.savefig(f'viewer-{epoch}.png')
