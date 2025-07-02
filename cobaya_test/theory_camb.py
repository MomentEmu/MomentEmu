import sys
import os
from MomentEmu import *
import camb

# Add this at the VERY TOP of theory_polycamb.py
sys.path.append("/Users/zzhang/Workspace/MomentEmu")

import numpy as np
from cobaya.theory import Theory
import pickle

# from cobaya.theories._base_classes import Theory

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

class rawCAMB(Theory):
    """
    Cobaya theory module using polynomial-based emulator for CMB D_ell.
    Converts D_ell → C_ell and returns {"tt": Cls}.
    """

    # Set ℓ_max of the emulator 
    ell_max_emulator = 2510
    ells = np.arange(2, ell_max_emulator + 1)
    ell_factors = 2 * np.pi / (ells * (ells + 1)) #/ (2.7255**2 * 1e12)    # muK2 --> FIRASK2 convention

    def initialize(self):
        self.pars = camb.CAMBparams()
        

    def get_requirements(self):
        return ['omega_b', 'omega_c', 'H0', 'logA', 'ns', 'tau'] 
    
    def get_Cl(self, lmax=2510, **kwargs):
        pars = camb.CAMBparams()
        params = self.provider

        omb = params.get_param("omega_b")
        omc = params.get_param("omega_c")
        H0 = params.get_param("H0")
        logA = params.get_param("logA")
        ns = params.get_param("ns")
        tau =params.get_param("tau")

        # Set cosmological parameters
        pars.set_cosmology(H0=H0, ombh2=omb, omch2=omc, tau=tau)

        # Initial power spectrum parameters
        As_arr = invert_log_As(logA)
        pars.InitPower.set_params(As=As_arr, ns=ns)

        # Compute lensed CMB power spectra up to lmax
        pars.set_for_lmax(lmax, lens_potential_accuracy=0)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')

        # Extract total TT spectrum (including lensing)
        CLtt = powers['total'][2: self.ell_max_emulator + 1, 0]
        CLtt *= self.ell_factors

        if lmax > self.ell_max_emulator:
            pad_len = lmax - self.ell_max_emulator
            CLtt = np.concatenate([CLtt, np.zeros(pad_len)])

        return {"tt": CLtt} 


    def get_can_provide(self):
        return {"Cl": ["tt"]}

    def must_provide(self, **_):
        return {}
