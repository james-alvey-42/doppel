import numpy as np
from scipy.stats import iqr
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon


def load_result_files(filepath_real, filepath_sim):
    result_real = np.load(filepath_real)
    result_sim = np.load(filepath_sim)
    print("Headers: C0 = p(x | real) / p(x | sim) \t p(x | sim) \t p(x | real)")
    return {"real": result_real, "sim": result_sim}


def compute_JSD(result1, result2, interpolation="quadratic", binning="FD"):
    if binning == "FD":
        number_of_bins_1 = int(
            (result1.max() - result1.min())
            / (2 * iqr(result1) * result1.size ** (-1 / 3))
        )
    elif binning == "S":
        number_of_bins_1 = int(1 + np.log2(result1.size))
    if number_of_bins_1 < 5:
        number_of_bins_1 = 5
    nbins1 = np.linspace(result1.min(), result1.max(), number_of_bins_1)

    if binning == "FD":
        number_of_bins_2 = int(
            (result2.max() - result2.min())
            / (2 * iqr(result2) * result2.size ** (-1 / 3))
        )
    elif binning == "S":
        number_of_bins_2 = int(1 + np.log2(result2.size))
    if number_of_bins_2 < 5:
        number_of_bins_2 = 5
    nbins2 = np.linspace(result2.min(), result2.max(), number_of_bins_2)
    h, bins = np.histogram(result1, density=True, bins=nbins1)
    bin_centres = 0.5 * (bins[1:] + bins[0:-1])

    min1, max1 = bin_centres[0], bin_centres[-1]
    fit1 = interp1d(
        bin_centres, h, kind=interpolation, fill_value=0.0, bounds_error=False
    )

    h, bins = np.histogram(result2, density=True, bins=nbins2)
    bin_centres = 0.5 * (bins[1:] + bins[0:-1])
    min2, max2 = bin_centres[0], bin_centres[-1]
    fit2 = interp1d(
        bin_centres, h, kind=interpolation, fill_value=0.0, bounds_error=False
    )
    x_grid = np.linspace(min(min1, min2), max(max1, max2), 1000)

    def fit_1(x):
        return max(1e-10, fit1(x))

    def fit_2(x):
        return max(1e-10, fit2(x))

    fit_1 = np.vectorize(fit_1)
    fit_2 = np.vectorize(fit_2)
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x_grid, fit_1(x_grid))
    plt.plot(x_grid, fit_2(x_grid))
    plt.show()

    JSD = jensenshannon(fit_1(x_grid), fit_2(x_grid)) ** 2
    return JSD, fit_1, fit_2


def jsd_factor(
    filepath_real, filepath_sim, interpolation="quadratic", binning="FD"
):
    results = load_result_files(
        filepath_real=filepath_real, filepath_sim=filepath_sim
    )
    result1 = np.log10(np.array(results["real"][:, 0]))
    result2 = np.log10(np.array(results["sim"][:, 0]))
    return compute_JSD(
        result1=result1,
        result2=result2,
        interpolation=interpolation,
        binning=binning,
    )
