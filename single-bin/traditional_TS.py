import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.optimize import brentq
from pathlib import Path


"""
This script is used to calculate the traditional test statistic (TS) for a single-bin counting experiment. The TS is
defined as the likelihood ratio between the signal+background and background-only hypotheses. The TS distribution is
used to calculate p-values and confidence intervals.

The script contains two models: model1 and model2. Model1 is the background-only hypothesis, and model2 is the
signal+background hypothesis. The TS is calculated using the likelihood ratio between model1 and model2.

The script also contains functions to calculate the p-value and confidence interval for a given TS value. The p-value
is calculated using the cumulative distribution function (CDF) of the TS distribution.

The script contains a main function that loops over different background rates and calculates the observed number of
events and TS value for each background rate. The results are saved to a file.

The script can be run from the command line with:
    python traditional_TS.py
This will calculate the TS value for a background rate of 10 and a signal strength of 1.
"""


def model1():
    bkg = np.random.poisson(10)
    return bkg


def model2(theta):
    # Here theta is the signal strength
    bkg = np.random.poisson(10)
    S = np.random.poisson(theta)
    return bkg + S


def TS(x, theta, bkg=10):
    """
    Note that this is independent of bkg so we can compute with fixed bkg
    """
    log_L1 = -bkg + x * np.log(bkg)
    log_L2 = -(bkg + theta) + x * np.log(bkg + theta)
    return -2 * (log_L1 - log_L2)


def get_Nobs(p_sig, bkg_):
    d = np.random.poisson(bkg_, 1000000)
    TS_vals = TS(d, 1, bkg=bkg_)
    critical_TS = np.quantile(TS_vals, [1 - p_sig])
    num_events = np.unique(d)
    TS_unique = np.unique(TS_vals)
    return num_events[np.where(TS_unique == critical_TS)][0]


def p_a(x0_, alpha, bkg_, plots=False):
    d = np.random.poisson(bkg_, 1000000)
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
    import sys

    bkgs = np.linspace(1, 100, 100)
    p_sig = float(sys.argv[1])
    N_obs_list = np.array([(get_Nobs(p_sig, bkg_) - bkg_) for bkg_ in bkgs])
    directory = Path(__file__).parent.resolve()
    np.save(
        str(Path(__file__).parent.resolve())
        + f"/output/traditional_ts_{p_sig}.npy",
        np.array([bkgs, N_obs_list]).T,
    )
    print(
        f"[{__file__.split('/')[-1]}] "
        + "Saved results to: "
        + str(Path(__file__).parent.resolve())
        + f"/output/traditional_ts_{p_sig}.npy"
    )
