import yaml
import os


def load_settings(config_path, model):
    if not os.path.exists(config_path):
        raise OSError(f"config file ({config_path}) does not exist")
    with open(config_path, "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    if settings["model"]["name"] != model:
        raise ValueError(
            f"Invalid model specified in config file ({config_path})"
        )
    print_settings(settings)
    return settings


def print_settings(settings):
    print("\n")
    for key in settings.keys():
        if type(settings[key]) == dict:
            print(f"{key}:")
            for k in settings[key].keys():
                if type(settings[key][k]) == dict:
                    print(f"  {k}:")
                    for kk in settings[key][k].keys():
                        print(
                            f"    {kk}: {settings[key][k][kk]} ({type(settings[key][k][kk]).__name__})"
                        )
                else:
                    print(
                        f"  {k}: {settings[key][k]} ({type(settings[key][k]).__name__})"
                    )

        else:
            print(f"{key}: {settings[key]} ({type(settings[key]).__name__})")
    print("\n")


def get_settings(settings):
    run_settings = settings.get(
        "run",
        {
            "verbose": False,
            "simulate": True,
            "train": False,
            "sample": False,
            "doppel": False,
        },
    )
    data_settings = settings.get(
        "data",
        {
            "store_name": "doppel_store",
            "store_size": 100_000,
            "chunk_size": 500,
        },
    )
    training_settings = settings.get(
        "training",
        {
            "trainer_dir": "doppel_training",
            "train_fraction": 0.85,
            "train_batch_size": 64,
            "val_batch_size": 64,
            "num_workers": 4,
            "device": "cpu",
            "n_devices": 1,
            "min_epochs": 1,
            "max_epochs": 50,
            "early_stopping_patience": 15,
            "learning_rate": 1e-4,
            "lr_scheduler_factor": 0.1,
            "lr_scheduler_patience": 3,
            "logger": {"type": None},
        },
    )
    sampling_settings = settings.get(
        "sampling",
        {"n_samples": 1000, "save_dir": "doppel_samples"},
    )
    doppel_settings = settings.get(
        "doppel",
        {
            "num_bins": 30,
            "interpolation": "linear",
            "plot": False,
            "plot_dir": ".",
        },
    )
    return (
        run_settings,
        data_settings,
        training_settings,
        sampling_settings,
        doppel_settings,
    )
