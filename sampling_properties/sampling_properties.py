import numpy as np
import sys
import doppel


class SingleBin(doppel.DoppelSim):
    def __init__(self, bkg=10, amp=1):
        super().__init__()
        self.bkg, self.amp = bkg, amp

    def generate_sim_data(self):
        bkg = np.float32(np.random.poisson(int(self.bkg), 1)) / self.bkg
        return bkg

    def generate_real_data(self):
        bkg = np.float32(np.random.poisson(self.bkg, 1)) / self.bkg
        sig = np.float32(np.random.poisson(self.amp, 1)) / self.bkg
        return bkg + sig


if __name__ == "__main__":
    model = "sampling_properties"
    config_file = sys.argv[1]
    settings = doppel.load_settings(config_file, model)

    simulator = SingleBin(
        bkg=settings["model"]["bkg"], amp=settings["model"]["amp"]
    )
    network = doppel.DoppelRatioEstimator(settings)

    doppel.simulate(simulator, settings)
    doppel.train(network, settings)
    doppel.sample(simulator, network, settings)
    doppel.doppel_factor(settings)
