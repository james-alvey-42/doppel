import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from scipy.special import rel_entr

# def jensen_shannon_divergence(samples1, samples2, nbins):
#     # Create a kernel density estimate of the distributions
#     kde1 = gaussian_kde(samples1)
#     kde2 = gaussian_kde(samples2)

#     # Evaluate KDE on a common support
#     xmin = min(samples1.min(), samples2.min())
#     xmax = max(samples1.max(), samples2.max())
#     x = np.linspace(xmin, xmax, nbins)

#     # Calculate the PDFs for both distributions on the common support
#     pdf1 = kde1(x)
#     pdf2 = kde2(x)
#     plt.plot(x, pdf1)
#     plt.plot(x, pdf2)
#     plt.savefig(f"{nbins}.png")

#     # Calculate the mid-point distribution (M)
#     m = (pdf1 + pdf2) / 2

#     # Calculate the Kullback-Leibler divergences
#     kl_div1 = rel_entr(pdf1, m)
#     kl_div2 = rel_entr(pdf2, m)

#     # Sum the KL divergences and take half
#     js_divergence = np.sum(kl_div1 + kl_div2) / 2

#     return js_divergence

def JSD_loop(data_size, n_trials):    
    

    # nbins_vals = np.linspace(50, 1000, 5)
    JSDs = []
    # for i in nbins_vals:
    for i in range(n_trials):

        # model_probabilities_saved_real = np.load(f"result-real-small.npy")
        model_probabilities_saved_real = np.load(f"result-real-{data_size}-{i}.npy")
        result1 = np.log10(model_probabilities_saved_real[:, 0])
        # model_probabilities_saved_sims = np.load(f"result-sims-small.npy")
        model_probabilities_saved_sims = np.load(f"result-sims-{data_size}-{i}.npy")
        result2 = np.log10(model_probabilities_saved_sims[:, 0])

        nbins = np.linspace(min(result1.min(), result2.min()), max(result1.max(), result2.max()), 100)

        # plt.figure(figsize=(7,5))
        h, bins = np.histogram(
            result1, density=True, bins=nbins
        )
        bin_centres = 0.5 * (bins[1:] + bins[0:-1])
        # plt.bar(
        #     bin_centres,
        #     h,
        #     width=bin_centres[1] - bin_centres[0],
        #     alpha=0.3,
        # )
        min1, max1 = bin_centres[0], bin_centres[-1]
        fit1 = interp1d(bin_centres, h, kind="cubic", fill_value="extrapolate")


        h, bins = np.histogram(
            result2, density=True, bins=nbins
        )
        bin_centres = 0.5 * (bins[1:] + bins[0:-1])
        # plt.bar(
        #     bin_centres,
        #     h,
        #     width=bin_centres[1] - bin_centres[0],
        #     alpha=0.3,
        # )
        min2, max2 = bin_centres[0], bin_centres[-1]
        fit2 = interp1d(bin_centres, h, kind="cubic", fill_value="extrapolate")

        x_grid = np.linspace(max(min1, min2), min(max1, max2), 1000)
        # plt.plot(x_grid, fit1(x_grid))
        # plt.plot(x_grid, fit2(x_grid))
        # plt.savefig("test_distributuons.png")

        def fit_1(x):
            return max(1e-10, fit1(x))

        def fit_2(x):
            return max(1e-10, fit2(x))

        fit_1 = np.vectorize(fit_1)
        fit_2 = np.vectorize(fit_2)

        JSD = jensenshannon(fit_1(x_grid), fit_2(x_grid)) ** 2

        # JSD = jensen_shannon_divergence(result1, result2, int(i))
        
        JSDs.append(JSD)

    plt.hist(JSDs, density=True)
    plt.xlabel("JSD")
    plt.savefig(f"JSD_distribution-{data_size}-{i}.png")


if __name__ == "__main__":
    import sys
    data_size = int(sys.argv[1])
    n_trials = int(sys.argv[2])
    JSD_loop(data_size, n_trials)