import numpy as np
import glob
import matplotlib.pyplot as plt


def parse_filename(fname):
    parts = fname.split("-")
    bkg = int(parts[1])
    amp = int(parts[2].split(".")[0])
    return bkg, amp


if __name__ == "__main__":
    files = glob.glob("result-*-*.npy")
    files.sort()
    p_sig = 0.32
    result_array = np.zeros((len(files), 3))
    for idx, file in enumerate(files):
        bkg, amp = parse_filename(file)
        result = np.load(file)
        result_array[idx, 0] = bkg
        critical_lr = np.quantile(result[:, 1], [1 - p_sig])
        logratios = np.unique(result[:, 1])
        total_events = np.unique(result[:, 0])
        critical_events = total_events[np.argmin(np.abs(logratios - critical_lr))]
        result_array[idx, 1] = critical_events - bkg
        result_array[idx, 2] = critical_lr
    np.save(f"output/doppel_result_{p_sig}.npy", result_array)
