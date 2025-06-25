import numpy as np
import pandas as pd
from tqdm import tqdm
from cosmopower import cosmopower_NN
from sklearn.model_selection import train_test_split

# Load pretrained emulator
cp_nn = cosmopower_NN(restore=True, 
    restore_filename='/path/to/cosmopower/trained_models/CP_paper/CMB/cmb_TT_NN')

# Parameter ranges (set based on CosmoPower training domain)
param_ranges = {
    'omega_b':     (0.0215, 0.0235),
    'omega_cdm':   (0.11,   0.125),
    'h':           (0.64,   0.76),
    'tau_reio':    (0.04,   0.09),
    'n_s':         (0.92,   1.00),
    'ln10^{10}A_s':(2.9,    3.2)
}

# Names (ordered)
param_names = list(param_ranges.keys())
n_params = len(param_names)

# Number of samples
N = 2000

# Use Latin Hypercube or uniform sampling
def sample_params(N, param_ranges):
    samples = np.random.rand(N, len(param_ranges))
    for i, (pname, (pmin, pmax)) in enumerate(param_ranges.items()):
        samples[:, i] = pmin + (pmax - pmin) * samples[:, i]
    return samples

# Generate parameter grid
theta_samples = sample_params(N, param_ranges)

# Convert to dict format for cosmopower
params_dict = {name: theta_samples[:, i].tolist() for i, name in enumerate(param_names)}

# Predict C_ell spectra: output is log10(C_ell), return 10^x
spectra = cp_nn.ten_to_predictions_np(params_dict)

# Optionally select subset of ell range (e.g., ell = 2 to 1500)
ell = np.arange(2, 2502)  # default in cosmopower

# Combine data for export or analysis
df_theta = pd.DataFrame(theta_samples, columns=param_names)
df_spectra = pd.DataFrame(spectra, columns=[f'C_ell_{l}' for l in ell])

# Save dataset
df_theta.to_csv('theta_samples.csv', index=False)
df_spectra.to_csv('spectra_samples.csv', index=False)

# Optionally merge into single file
df_all = pd.concat([df_theta, df_spectra], axis=1)
df_all.to_csv('cmb_tt_dataset.csv', index=False)

print("✅ Dataset saved: cmb_tt_dataset.csv")
