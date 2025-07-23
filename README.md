# MomentEmu

A lightweight, interpretable polynomial emulator for smooth mappings, implemented in pure Python.


## 📖 Overview

**MomentEmu.py** implements the **moment-projection polynomial emulator** introduced in Zhang (2025) ([arXiv:2507.02179](https://arxiv.org/abs/2507.02179)).

It builds interpretable, closed-form polynomial emulators via moment matrices, achieving millisecond-level inference and symbolic transparency. Two demonstration models include:

- **PolyCAMB‑Dℓ**: Maps six cosmological parameters to the CMB temperature power spectrum (accuracy ~0.03% for ℓ ≤ 2510).
- **PolyCAMB‑peak**: Supports both forward and inverse mapping between parameters and acoustic peak features (sub-percent accuracy).

## 🚀 Features

- Pure Python implementation; minimal dependencies (`numpy`, `scipy`, `sympy`, ...)
- Closed-form polynomial expressions (symbolic)
- Supports **forward** (θ → y) and **inverse** (y → θ) emulation
- Fast training via moment matrices; near-instant inference
- Suitable for MCMC, Bayesian inference, sensitivity analyses, and gradient-based tasks
- Compact—no heavy model files

## 🛠️ Installation

```bash
git clone https://github.com/zzhang0123/MomentEmu.git
cd MomentEmu
```

## 🧪 Examples

- `PolyCAMB.py`: Source code for PolyCAMB-Dℓ and PolyCAMB-peak emulators.
- `notebooks/poly_camb.ipynb`: Builds the PolyCAMB‑Dℓ emulator, as well as both forward and inverse PolyCAMB‑peak models.

Below is an abstract example usage.

```python

from MomentEmu import PolyEmu

# Define placeholder training data (shapes: X (N, n), Y (N, m))
X = ...  # e.g., parameter samples
Y = ...  # e.g., observable outputs

# Emulator training and prediction
emulator = PolyEmu(X, Y, forward=True, backward=True)    # Set backward=False for forward-only emulation

# Forward-mode emulator prediction
X_new = ...       # new input sample(s) (shape (n,) or (k, n))
Y_pred = emulator.forward_emulator(X_new)

# Backward (inverse) emulator inference
Y_new = ...       # new output sample(s) (shape (m,) or (k, m))
X_est = emulator.backward_emulator(Y_new)

# Retrieve symbolic polynomial expressions for each output/input dimension
sym_fwd = emulator.generate_forward_symb_emu(variable_names=None)
print(sym_fwd)
sym_bwd = emulator.generate_backward_symb_emu(variable_names=None)
print(sym_bwd)

# For a full working example with real data, see notebooks/poly_camb.ipynb

```

---

### 🧠 How It Works

MomentEmu builds:

- A **moment matrix**  
  $M_{\alpha\beta} = \frac{1}{N} \sum_{i} \theta_i^\alpha\, \theta_i^\beta$

- A **moment vector**  
  $\nu_\alpha = \frac{1}{N} \sum_{i} \theta_i^\alpha\, y_i$

Solving $M c = \nu$ finds polynomial coefficients $c$. No iterative optimization is needed --- model selection uses validation RMSE.  
[Read more in the arXiv paper](https://arxiv.org/abs/2507.02179). 


