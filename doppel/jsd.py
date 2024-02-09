import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import matplotlib.pyplot as plt


def compute_doppel_factor(
    result1,
    result2,
    interpolation="linear",
    nbins=40,
    plot=False,
    plot_dir=".",
):
    xmin = min(result1.min(), result2.min())
    xmax = max(result1.max(), result2.max())
    x_grid = np.linspace(xmin, xmax, 1000)

    x_bins = np.linspace(
        min(result1.min(), result2.min()),
        max(result1.max(), result2.max()),
        nbins,
    )
    x_bin_centres = 0.5 * (x_bins[1:] + x_bins[0:-1])

    h1, _ = np.histogram(result1, density=True, bins=x_bins)

    h2, _ = np.histogram(result2, density=True, bins=x_bins)

    h1[h1 == 0] = 1e-200
    h2[h2 == 0] = 1e-200

    fitkl = interp1d(
        x_bin_centres,
        (h1 * (np.log(h1) - np.log(0.5 * (h1 + h2))))
        + (h2 * (np.log(h2) - np.log(0.5 * (h1 + h2)))),
        kind=interpolation,
        fill_value=0.0,
        bounds_error=False,
    )

    def fit_kl(x):
        return max(1e-100, fitkl(x))

    fit_kl = np.vectorize(fit_kl)
    if plot:
        plt.figure(figsize=(12, 6))
        ax = plt.subplot(1, 2, 1)
        plt.hist(
            result1, bins=x_bins, density=True, histtype="step", color="#648FFF"
        )
        plt.hist(
            result2, bins=x_bins, density=True, histtype="step", color="#DC267F"
        )
        plt.hist(result1, bins=x_bins, density=True, alpha=0.2, color="#648FFF")
        plt.hist(
            result2,
            bins=x_bins,
            density=True,
            alpha=0.2,
            color="#DC267F",
        )
        plt.xlabel(r"$\log_{10} C_0(x)$")
        ax = plt.subplot(1, 2, 2)
        plt.plot(x_grid, fit_kl(x_grid) * 0.5)
        plt.xlabel(r"$\log_{10} C_0(x)$")
        plt.ylabel(r"$\mathrm{d}JSD/\mathrm{d}\log_10 C_0(x)$")
        plt.tight_layout()
        plt.savefig(plot_dir + "/doppel_factor.pdf")
    return 0.5 * simpson(y=fit_kl(x_grid), x=x_grid)
