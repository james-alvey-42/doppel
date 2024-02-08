import numpy as np
import swyft

if swyft.__version__ != "0.4.5":
    raise ImportError("ERROR: swyft version must be 0.4.5")


class DoppelSim(swyft.Simulator):
    def __init__(self):
        super().__init__()
        self.transform_samples = swyft.to_numpy32

    def sample_model(self):
        return np.array([np.random.choice(2)])

    def generate_sim_data(self):
        raise NotImplementedError(
            "generate_sim_data method must be implemented"
        )

    def generate_real_data(self):
        return NotImplementedError(
            "generate_real_data method must be implemented"
        )

    def sample_data(self, model):
        if model == 0:
            return self.generate_sim_data()
        elif model == 1:
            return self.generate_real_data()
        else:
            err_msg = f"Model must be in [0, 1] | model = {model}"
            raise ValueError(err_msg)

    def build(self, graph):
        model = graph.node("model", self.sample_model)
        data = graph.node("data", self.sample_data, model)
