import numpy as np
import sys
import torch
import swyft
import doppel


class GravitationalWaves(doppel.DoppelSim):
    def __init__(self, fmin=3e-1, fmax=1e2, npts=10):
        super().__init__()
        self.f_array = np.geomspace(fmin, fmax, npts)

    def generate_sim_data(self):
        return np.random.normal(0.0, np.sqrt(self.sim_PSD(self.f_array)))

    def generate_real_data(self):
        return np.random.normal(0.0, np.sqrt(self.real_PSD(self.f_array)))

    def sim_PSD(self, f, normalization=1.0):
        return (
            normalization
            * 9
            * ((4.49 * f) ** (-56) + 0.16 * f ** (-4.52) + 0.52 + 0.32 * f**2)
        )

    def real_PSD(self, f, normalization=1.0):
        return (
            normalization
            * 11.0
            * (
                (4.49 * f) ** (-57)
                + 0.2 * f ** (-4.7)
                + 0.56
                + 0.39 * f**1.75
            )
        )


class LinearCompression(torch.nn.Module):
    def __init__(self, hidden_size=10, num_features=1):
        super(LinearCompression, self).__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.LazyLinear(hidden_size),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(hidden_size),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(num_features),
        )

    def forward(self, x):
        return self.sequential(x)


class GWNetwork(doppel.DoppelRatioEstimator):
    def __init__(self, settings):
        super().__init__(settings)
        self.online_normalisation = swyft.networks.OnlineStandardizingLayer(
            shape=torch.Size([settings["model"]["npts"]])
        )
        self.linear_compression = LinearCompression(
            hidden_size=settings["model"].get("hidden_size", 10),
            num_features=settings.get("training", {}).get("num_features", 1),
        )

    def compression(self, data):
        norm_data = self.online_normalisation(data)
        return self.linear_compression(norm_data)


if __name__ == "__main__":
    model = "gravitational_waves"
    config_file = sys.argv[1]
    settings = doppel.load_settings(config_file, model)

    simulator = GravitationalWaves(
        fmin=settings["model"]["fmin"],
        fmax=settings["model"]["fmax"],
        npts=settings["model"]["npts"],
    )
    network = GWNetwork(settings)

    doppel.simulate(simulator, settings)
    doppel.train(network, settings)
    doppel.sample(simulator, network, settings)
    doppel.doppel_factor(settings)
