# DOPPEL
A Bayesian method for quantifying the goodness-of-fit of a model/simulator compared to real data without a likelihood

### Results Generation

1. Single-bin Experiment
    - `traditional_TS.py`: script to generate the relevant observed numbers of events required to reject the backgorund-only hypothesis. Run using `python traditional_TS.py p_sig` where `p_sig` is (1 -) confidence level. Generates output in the folder `output/traditional_ts_[p_sig].npy` with columns [`background rate`, `number of events required`].
    - Use `run_traditional_ts.sh` to generate the relevant data for Fig. 1 in the corresponding paper.
