import sys
import os
from MomentEmu import *

# Add this at the VERY TOP of theory_polycamb.py
sys.path.append("/Users/zzhang/Workspace/MomentEmu")

import numpy as np
from cobaya.theory import Theory
import pickle

# from cobaya.theories._base_classes import Theory


# class PolyCAMB(Theory):
#     """
#     Cobaya theory module using polynomial-based emulator for CMB D_ell.
#     Converts D_ell → C_ell and returns {"tt": Cls}.
#     """

#     # Set ℓ_max of the emulator (e.g., 2050)
#     ell_max_emulator = 2050
#     ells = np.arange(2, ell_max_emulator + 1)
#     ell_factors = 2 * np.pi / (ells * (ells + 1)) / (2.7255**2)    # muK2 --> FIRASmuK2 convention

#     def initialize(self):
#         # Load the trained emulator (update filename if needed)
#         with open("emulators/PolyCAMB_Dl.pkl", "rb") as f:
#             self.emu = pickle.load(f)

#         # Precompute ℓ values
#         # self.ells = np.arange(2, self.ell_max_emulator + 1)
#         # self.ell_factor = 2 * np.pi / (self.ells * (self.ells + 1))

#     def get_requirements(self):
#         return ['omega_b', 'omega_c', 'H0', 'logA', 'ns', 'tau']

#     def get_Cl(self, ell_max=2510, **kwargs):
#         # Limit to emulator range
#         # ell_cap = min(ell_max, self.ell_max_emulator)
        

#         # Collect input parameters
#         params = self.provider


#         theta = np.array([
#             params.get_param("omega_b"),
#             params.get_param("omega_c"),
#             params.get_param("H0"),
#             params.get_param("logA"),
#             params.get_param("ns"),
#             params.get_param("tau"),
#         ])

#         # Predict D_ell from emulator
#         D_ell_pred = self.emu.forward_emulator(theta.reshape(1, -1)).flatten()[2: self.ell_max_emulator + 1]

#         # Convert to C_ell: C = D * 2π / ℓ(ℓ+1)
#         C_ell_pred = D_ell_pred * self.ell_factors 

#         # Pad with zeros if ell_max > emulator limit
#         if ell_max > self.ell_max_emulator:
#             pad_len = ell_max - self.ell_max_emulator
#             C_ell_pred = np.concatenate([C_ell_pred, np.zeros(pad_len)])

#         print(f"First 5 C_ℓ: {C_ell_pred[:5]}")  # Should be ~5000-6000
#         # Return as required dictionary
#         return {"tt": C_ell_pred}

#     def get_can_provide(self):
#         return {"Cl": ["tt"]}

#     def must_provide(self, **_):
#         return {}

from cobaya.conventions import outputs_CMB

class PolyCAMB(Theory):
    """A Cobaya *theory* plugin that supplies Cℓ^TT from a polynomial‑based
    D_ℓ emulator.

    ‑‑ Usage in YAML ‑‑‑‑
    theory:
      camb:          # optional – keep for TE/EE/φφ or remove if your emulator also covers them
      polycamb:
        module: poly_camb      # file name without .py
        class: PolyCAMB
        emulator_path: emulators/PolyCAMB_Dl.pkl  # change if different
        lmax: 2508             # must be ≥ Planck high‑ℓ cut‑off
    """

    # Default initialisation values can be overridden from YAML
    def initialize(self, emulator_path: str = "emulators/PolyCAMB_Dl_N6.pkl", lmax: int = 2508):
        """Load the trained emulator and pre‑compute conversion factors.

        Parameters
        ----------
        emulator_path : str
            Path to your pickled emulator (must have a ``forward_emulator`` method).
        lmax : int
            Largest multipole that will ever be requested by the likelihood.
        """
        self.lmax = lmax

        # ---- load emulator ----
        with open(emulator_path, "rb") as f:
            self.emu = pickle.load(f)

        # Emulator may have been trained only up to a smaller ℓ_max
        try:
            # assume the object knows its own limit
            self.ell_max_emulator = self.emu.ell_max  # type: ignore[attr-defined]
        except AttributeError:
            # fallback: trust user to set lmax ≤ training range
            self.ell_max_emulator = lmax

        # ℓ array starting at 2 (ℓ=0,1 unused by Planck)
        self._ells = np.arange(2, self.ell_max_emulator + 1)
        # conversion D_ℓ [µK²] → C_ℓ [µK²]
        self._dl_to_cl = 2 * np.pi / (self._ells * (self._ells + 1))

    # ------------------------------------------------------------------
    #  Cobaya plumbing: advertise what we can supply & our dependencies.
    # ------------------------------------------------------------------
    def get_can_provide(self):
        """We provide the full CMB Cℓ dictionary (only TT non‑zero)."""
        return [outputs_CMB]

    def get_requirements(self):
        """Our prediction uses the six ΛCDM parameters; we require *no* other
        theory input."""
        return {}

    # ------------------------------------------------------------------
    #  Core computation – called every time Cobaya evaluates the likelihood
    # ------------------------------------------------------------------
    def calculate(self, state, want_cl=None, **kwargs):
        """Compute Cℓ^TT up to *self.lmax* and store in ``self.current_state``.

        Cobaya merges spectra from multiple *theory* blocks, so we only fill
        the TT field, leaving TE/EE/etc. for CAMB (if present) to provide.
        """
        # Parameter names **must match** those declared in your YAML *params*.
        p = state["params"]
        theta = np.array([
            p["omega_b"],   # Ω_b h²
            p["omega_c"],   # Ω_c h² (rename in YAML if you use omega_cdm)
            p["H0"],        # H₀ [km s⁻¹ Mpc⁻¹]
            p["logA"],      # ln(10¹⁰ A_s) or log(10¹⁰ A_s) — consistent with training!
            p["ns"],        # n_s
            p["tau"],       # τ_reio
        ])

        # ---- Emulator predicts D_ℓ (µK²) starting at ℓ=0 or 2? ----
        Dl = self.emu.forward_emulator(theta.reshape(1, -1)).flatten()
        Dl = Dl[2 : self.ell_max_emulator + 1]  # keep ℓ≥2

        # Convert to C_ℓ and pad up to *self.lmax*
        Cl_tt = np.zeros(self.lmax + 1)  # indices 0 … lmax
        Cl_tt[2 : self.ell_max_emulator + 1] = Dl * self._dl_to_cl

        # if lmax requested by likelihood > emulator limit, leave rest at 0
        # (Planck high‑ℓ TT stops at ℓ=2508, so impact negligible)

        # Store for the likelihood – only TT provided.
        self.current_state["Cl"] = {"TT": Cl_tt}
