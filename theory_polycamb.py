import sys
import os
from MomentEmu import *

# Add this at the VERY TOP of theory_polycamb.py
sys.path.append("/Users/zzhang/Workspace/MomentEmu")

import numpy as np
from cobaya.theory import Theory
import pickle

# from cobaya.theories._base_classes import Theory


class PolyCAMB(Theory):
    """
    Cobaya theory module using polynomial-based emulator for CMB D_ell.
    Converts D_ell → C_ell and returns {"tt": Cls}.
    """

    # Set ℓ_max of the emulator 
    ell_max_emulator = 2510
    ells = np.arange(2, ell_max_emulator + 1)
    ell_factors = 2 * np.pi / (ells * (ells + 1)) #/ (2.7255**2 * 1e12)    # muK2 --> FIRASK2 convention

    def initialize(self):
        # Load the trained emulator (update filename if needed)
        with open("emulators/PolyCAMB_Dl_N7.pkl", "rb") as f:
            self.emu = pickle.load(f)

    def get_requirements(self):
        return ['omega_b', 'omega_c', 'H0', 'logA', 'ns', 'tau']

    def get_Cl(self, ell_max=2510, **kwargs):
        # Limit to emulator range
        # ell_cap = min(ell_max, self.ell_max_emulator)
        

        # Collect input parameters
        params = self.provider


        theta = np.array([
            params.get_param("omega_b"),
            params.get_param("omega_c"),
            params.get_param("H0"),
            params.get_param("logA"),
            params.get_param("ns"),
            params.get_param("tau"),
        ])

        # Predict D_ell from emulator
        D_ell_pred = self.emu.forward_emulator(theta.reshape(1, -1)).flatten()[2: self.ell_max_emulator + 1]

        # Convert to C_ell: C = D * 2π / ℓ(ℓ+1)
        C_ell_pred = D_ell_pred * self.ell_factors 

        # Pad with zeros if ell_max > emulator limit
        if ell_max > self.ell_max_emulator:
            pad_len = ell_max - self.ell_max_emulator
            C_ell_pred = np.concatenate([C_ell_pred, np.zeros(pad_len)])

        # Return as required dictionary
        return {"tt": C_ell_pred}

    def get_can_provide(self):
        return {"Cl": ["tt"]}

    def must_provide(self, **_):
        return {}
