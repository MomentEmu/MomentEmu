import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from pyDOE import lhs 
import camb
from camb import model, initialpower
from tqdm import tqdm
import pandas as pd
from itertools import product


# Define parameter bounds
demo_bounds = {
    'ombh2':   (0.019, 0.025),
    'omch2':   (0.09, 0.15),
    'H0':      (55.0, 80.0),
    'ln_1e10_As':(2.7, 3.2),
    'ns':      (0.88, 1.02),
    'tau':     (0.02, 0.12)
    }

def invert_log_As(ln_1e10_As):
    """
    Invert the logarithmic scaling of the primordial amplitude.
    Parameters:
        ln_1e10_As : float
            Logarithm of the primordial amplitude in units of 1e-10 As.
    Returns:
        As : float
            Primordial amplitude in units of As.
    """
    As = np.exp(ln_1e10_As) * 1e-10
    return As

def sample_lcdm_params_grid(n_grid_per_param=5, return_DataFrame=False, bound=None):
    """
    Generate a Cartesian product grid over ΛCDM parameters.

    Parameters:
        n_grid_per_param : int
            Number of grid points per parameter (uniform for all).
        return_DataFrame : bool
            If True, return a dictionary of arrays by parameter name.

    Returns:
        grid_samples : array, shape (N_total, 6)
            Grid samples for each parameter combination.
        param_names : list
            List of parameter names in order.
    """

    

    if bound is not None:
        param_bounds = bound
    else:
        param_bounds = demo_bounds
    
    param_names = list(param_bounds.keys())

    # Create grid arrays for each parameter
    param_grids = []
    for name in param_names:
        low, high = param_bounds[name]
        grid = np.linspace(low, high, n_grid_per_param, endpoint=True)
        param_grids.append(grid)

    # Cartesian product of grid points
    all_combinations = list(product(*param_grids))
    grid_array = np.array(all_combinations)

    # # Convert ln_1e10_As → As
    # ln10_As = grid_array[:, param_names.index('ln_1e10_As')]
    # As = np.exp(ln10_As) * 1e-10
    # grid_array[:, param_names.index('ln_1e10_As')] = As

    if return_DataFrame:
        param_array = {
            name: grid_array[:, i] for i, name in enumerate(param_names)
        }
        # param_array['As'] = param_array.pop('ln_1e10_As')  # rename
        return pd.DataFrame(param_array)

    # param_names[3] = 'As'
    return grid_array, param_names

def sample_lcdm_params_rand(n_samples=1000, seed=None, return_DataFrame=False, use_lhs=False, bound=None):
    np.random.seed(seed)

    if bound is not None:
        param_bounds = bound
    else:
        param_bounds = demo_bounds

    param_names = list(param_bounds.keys())
    ndim = len(param_names)

    if use_lhs:
        # Latin Hypercube sampling in unit cube
        unit_samples = lhs(ndim, samples=n_samples, criterion='maximin')

        # Scale to parameter bounds
        scaled_samples = np.zeros_like(unit_samples)
        for i, name in enumerate(param_names):
            low, high = param_bounds[name]
            scaled_samples[:, i] = low + (high - low) * unit_samples[:, i]
    else:
        # Uniform random sampling
        scaled_samples = np.zeros((n_samples, ndim))
        for i, name in enumerate(param_names):
            low, high = param_bounds[name]
            scaled_samples[:, i] = np.random.uniform(low, high, size=n_samples)

    # # Convert ln_1e10_As to As
    # ln10_As = scaled_samples[:, param_names.index('ln_1e10_As')]
    # As = np.exp(ln10_As) * 1e-10
    # scaled_samples[:, 3] = As

    if return_DataFrame:
        # Create final parameter array
        param_array = {
            'ombh2':        scaled_samples[:, 0],
            'omch2':        scaled_samples[:, 1],
            'H0':           scaled_samples[:, 2],
            'ln_1e10_As':   scaled_samples[:, 3],
            'ns':           scaled_samples[:, 4],
            'tau':          scaled_samples[:, 5]
        }
        return pd.DataFrame(param_array)

    # param_names[3] = 'As'
    return scaled_samples, param_names


def get_aps(theta, lmax=2000):
    pars = camb.CAMBparams()

    # Set cosmological parameters
    pars.set_cosmology(H0=theta['H0'], ombh2=theta['ombh2'], omch2=theta['omch2'], tau=theta['tau'])

    # Initial power spectrum parameters
    As_arr = invert_log_As(theta['ln_1e10_As'])
    pars.InitPower.set_params(As=As_arr, ns=theta['ns'])

    # Compute lensed CMB power spectra up to lmax
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')

    # Extract total TT spectrum (including lensing)
    totCL = powers['total']
    ell = np.arange(totCL.shape[0])

    return ell, totCL[:, 0]  # Return (ℓ, C_ℓ^TT)


def generate_dataset(N, lmax=3000, ell_sampling=None, grid=True, bound=None):
    """
    Generate a dataset of CMB power spectra using CAMB.
    Parameters:
    - N: number of simulations if grid=False, otherwise number per dimmension.
    - lmax: maximum multipole to compute
    - ell_sampling: 'log' for log-sampling ℓ, 'linear' for linear sampling
    - grid: if True, use grid sampling, otherwise use random sampling.
    Returns:
    - X: parameters (N x 6)
    - Y: CMB power spectra (N x len(ells))
    - param_names: list of parameter names
    - ell_sampled: sampled ℓ values
    """
    if grid:
        params = sample_lcdm_params_grid(n_grid_per_param=N, return_DataFrame=True, bound=None)
        N = len(params)
    else:
        params = sample_lcdm_params_rand(N, seed=42, return_DataFrame=True, bound=None)

    # Optional: log-sample ℓ to reduce data dimensionality
    if ell_sampling == 'log':
        ell_sampled = np.unique(np.round(np.exp(np.linspace(np.log(2), np.log(lmax), 200))).astype(int))
        ell_sampled = ell_sampled[ell_sampled <= lmax]

    Cls_all = []

    for i in tqdm(range(N), desc="Running CAMB"):
        theta = params.iloc[i]
        ell, Cl_tt = get_aps(theta, lmax=lmax)
        if ell_sampling is not None:
            # Interpolate to sampled ℓs
            Cl_sampled = np.interp(ell_sampled, ell, Cl_tt)
            Cls_all.append(Cl_sampled)
        else:
            ell_sampled = ell
            Cls_all.append(Cl_tt)

    # Final array: (N samples) x (len(ell_sampled) features)
    X = params.to_numpy()
    Y = np.array(Cls_all)
    param_names = params.columns.tolist()

    return X, Y, param_names, ell_sampled

### Functions for finding features of the angular power spectra ###

def find_aps_peaks(ell, Cl, smooth_sigma=1.5, height=None, distance=20):
    """
    Find peak positions and heights in a TT CMB power spectrum.

    Parameters:
    - ell: array of multipoles (ℓ)
    - Cl: array of C_ell values (same length as ell)
    - smooth_sigma: optional Gaussian smoothing width (in pixels)
    - height: minimum height of peak (can be scalar or tuple for range)
    - distance: minimum distance between peaks (in ℓ pixels)

    Returns:
    - ell_peaks: ℓ values at peaks
    - Cl_peaks: corresponding C_ell values at peaks
    """

    # Step 1: Optional smoothing
    Cl_smooth = gaussian_filter1d(Cl, sigma=smooth_sigma)

    # Step 2: Find peaks
    peak_indices, _ = find_peaks(Cl_smooth, height=height, distance=distance)

    ell_peaks = ell[peak_indices]
    Cl_peaks = Cl[peak_indices]

    return ell_peaks, Cl_peaks

def compute_aps_moments(ell, Cl, max_order=6, normalize=True):
    """
    Compute the first K monomial moments of the angular power spectrum C_ell.

    Parameters:
        ell : array-like, shape (N,)
            The multipole values (e.g., 2 to 2500)
        Cl : array-like, shape (N,)
            The angular power spectrum values (same length as ell)
        max_order : int
            Maximum moment order to compute (inclusive)
        normalize : bool
            If True, normalize weights so moments are dimensionless

    Returns:
        moments : array, shape (max_order + 1,)
            Monomial moments mu_k = sum ell^k * C_ell * w(ell)
    """
    ell = np.asarray(ell)
    Cl = np.asarray(Cl)
    w = np.ones_like(ell)

    if normalize:
        # Normalize weights so that sum(w * Cl) = 1
        Z = np.sum(w * Cl)
        if Z == 0:
            raise ValueError("Sum of C_ell is zero — cannot normalize")
        w = w / Z

    # Compute moments: mu_k = sum ell^k * C_ell * w
    moments = []
    for k in range(max_order + 1):
        mu_k = np.sum((ell ** k) * Cl * w)
        moments.append(mu_k)

    return np.array(moments)

def peak_summary(ells, D_ell_samples, n_peaks, tilt=True, fractional_height=True):
    peak_part1 = []
    peak_part2 = []

    n_samples = D_ell_samples.shape[0]
    for i in tqdm(range(n_samples)):
        ell_peaks, hei_peaks = find_aps_peaks(ells[50:], D_ell_samples[i, 50:], smooth_sigma=1, height=None, distance=2)
        ell_peaks = ell_peaks[:n_peaks]
        hei_peaks = hei_peaks[:n_peaks]
        if tilt:
            peak_part1.append(hei_peaks/ell_peaks)
        else:
            peak_part1.append(ell_peaks)
        if fractional_height:
            hei_peaks[1:] /= hei_peaks[0]
        peak_part2.append(hei_peaks)

    peak_part1 = np.array(peak_part1)
    peak_part2 = np.array(peak_part2)

    peak_arr = np.hstack((peak_part1, peak_part2))
    return peak_arr

def moment_summary(ells, D_ell_samples, max_order):
    moment_li = []
    n_samples = D_ell_samples.shape[0]
    for i in tqdm(range(n_samples)):
        moment_li.append(compute_aps_moments(ells, D_ell_samples[i], max_order=max_order))
    return np.array(moment_li)

def thinned_Dell(ells,  D_ell_samples, n_ell_sampled, init_ell=50):
    '''
    thin the D_ell_samples from (N_samples, N_ell) to (N_samples, N_ell_sampled) with roughly uniform spacing
    '''
    inds = np.linspace(init_ell, len(ells), n_ell_sampled)
    thinned_D_ell_samples = D_ell_samples[:, inds.astype(int)]
    return thinned_D_ell_samples

def peak_pos_hei(peak_arr):
    N1, N2 = peak_arr.shape
    Npeaks = N2//2
    peak_tilts = peak_arr[:, :Npeaks]
    peak_heights = peak_arr[:, Npeaks:]
    for j in range(peak_heights.shape[0]):
        peak_heights[j, 1:] *= peak_heights[j, 0]
    peak_ells = peak_heights/peak_tilts
    peak_ells = peak_ells.astype(int)
    return peak_ells, peak_heights

def peak2_over_peak1(ombh2, omch2, ns):
    om_m = ombh2 + omch2
    numer =  0.925 * om_m ** 0.18 * 2.4 ** (ns-1) 
    denomi = (1 + (ombh2/0.0164)**(12 * om_m**0.52))**0.2
    return numer / denomi

def peak3_over_peak1(ombh2, omch2, ns):
    om_m = ombh2 + omch2
    numer =  2.17 * om_m ** 0.59 * 3.6 ** (ns-1) 
    denomi = (1 + (ombh2/0.044)**2) * ( 1 + 1.63 * (1 - ombh2 / 0.071) * om_m )
    return numer / denomi

