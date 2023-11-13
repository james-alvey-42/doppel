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
    #prior_samples = swyft.Samples({"model": np.array([[0], [1]])})

    print("Simulated data generation")
    Nsim = 100000
    A_sim = simulator.sample(Nsim,
            targets=["data"], conditions={"model": np.array([0])}
        )
    print("Real data generation")
    A_real = simulator.sample(Nsim,
            targets=["data"], conditions={"model": np.array([1])}
        )

    network.eval()
    with torch.no_grad():
        lp_sim_0 = np.array(network.forward({'data': torch.tensor(A_sim['data'])}, {'model': torch.tensor([[0]])}).logratios).T[0]
        lp_sim_1 = np.array(network.forward({'data': torch.tensor(A_sim['data'])}, {'model': torch.tensor([[1]])}).logratios).T[0]
        lp_real_0 = np.array(network.forward({'data': torch.tensor(A_real['data'])}, {'model': torch.tensor([[0]])}).logratios).T[0]
        lp_real_1 = np.array(network.forward({'data': torch.tensor(A_real['data'])}, {'model': torch.tensor([[1]])}).logratios).T[0]
    
    model_probabilities_saved_sims = np.zeros((Nsim, 3))
    model_probabilities_saved_real = np.zeros((Nsim, 3))

    model_probabilities_saved_sims[:, -2] = np.exp(lp_sim_0 )/ (np.exp(lp_sim_0 )+ np.exp(lp_sim_1))
    model_probabilities_saved_sims[:, -1] = np.exp(lp_sim_1 )/ (np.exp(lp_sim_0 )+ np.exp(lp_sim_1))
    model_probabilities_saved_sims[:, 0] = model_probabilities_saved_sims[:, -1] / model_probabilities_saved_sims[:, -2] 

    model_probabilities_saved_real[:, -2] = np.exp(lp_real_0) / (np.exp(lp_real_0) + np.exp(lp_real_1))
    model_probabilities_saved_real[:, -1] = np.exp(lp_real_1) / (np.exp(lp_real_0) + np.exp(lp_real_1))
    model_probabilities_saved_real[:, 0] = model_probabilities_saved_real[:, -1] / model_probabilities_saved_real[:, -2] 

    np.save(
        f"result-sims-{epoch}.npy",
        np.array(model_probabilities_saved_sims),
    )
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

    
