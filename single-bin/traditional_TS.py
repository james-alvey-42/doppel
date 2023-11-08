import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.optimize import brentq
from pathlib import Path

palette = ["#2e4854", "#557b82", "#bab2a9", "#c98769", "#a1553a"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=palette)


def model1():
    bkg = np.random.poisson(10)
    return bkg


def model2(theta):
    # Here theta is the signal strength
    bkg = np.random.poisson(10)
    S = np.random.poisson(theta)
    return bkg + S


def TS(x, theta, bkg=10):
    log_L1 = -bkg + x * np.log(bkg)
    log_L2 = -(bkg + theta) + x * np.log(bkg + theta)
    return -2 * (log_L1 - log_L2)

def get_Nobs(p_sig, bkg_):
    d = np.random.poisson(bkg_, 100000)
    TS_vals = TS(d, 1, bkg=bkg_)
    critical_TS = np.quantile(TS_vals, [1 - p_sig])
    num_events = np.unique(d)
    TS_unique = np.unique(TS_vals)
    return num_events[np.where(TS_unique == critical_TS)][0]


def p_a(x0_, alpha, bkg_, plots=False):
    d = np.random.poisson(bkg_, 100000)
    TS_vals = TS(d, alpha, bkg=bkg_)
    P_TS = gaussian_kde(TS_vals)
    TS_obs = TS(x0_, alpha, bkg=bkg_)
    if plots:
        x = np.linspace(-50, 20, 1000)
        plt.axvline(x=TS_obs, color="k", linestyle="--")
        plt.plot(x, P_TS.pdf(x))
        plt.xlim(x.min(), x.max())
        plt.show()
    return P_TS.integrate_box_1d(TS_obs, np.inf)


def find_Nobs(p_sig, bkg_):
    def min_func(x0_):
        return p_a(x0_, 1, bkg_) - p_sig

    return brentq(min_func, 0, 500)


if __name__ == "__main__":
    bkgs = np.linspace(1, 100, 100)
    p_sig = 0.32
    #N_obs_list = np.array([(find_Nobs(0.01, bkg_) - bkg_) for bkg_ in bkgs])
    N_obs_list = np.array([(get_Nobs(p_sig, bkg_) - bkg_) for bkg_ in bkgs])
    np.save(f"output/traditional_ts_{p_sig}.npy", np.array([bkgs, N_obs_list]).T)
