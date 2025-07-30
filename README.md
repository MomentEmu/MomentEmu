# MomentEmu

A lightweight, interpretable polynomial emulator for smooth mappings, implemented in pure Python.


## 📖 Overview

**MomentEmu** implements the **moment-projection polynomial emulator** introduced in Zhang (2025) ([arXiv:2507.02179](https://arxiv.org/abs/2507.02179)).
It builds interpretable, closed-form polynomial emulators via moment matrices, achieving millisecond-level inference and symbolic transparency.

For a complete working example demonstrating MomentEmu applied to cosmological parameter estimation (PolyCAMB), see the companion repository: [MomentEmu-PolyCAMB-examples](https://github.com/zzhang0123/MomentEmu-PolyCAMB-examples).

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

Or install directly:
```bash
pip install git+https://github.com/zzhang0123/MomentEmu.git
```

## 📋 Dependencies

- `numpy`
- `scipy` 
- `sympy`
- `scikit-learn`

## 🧪 Quick Start

**Note**: Make sure to install MomentEmu first using one of the installation methods above.

```python
from MomentEmu import PolyEmu

# Define your training data
X = ...  # Input parameters, shape (N, n)
Y = ...  # Output observables, shape (N, m)

# Create emulator with both forward and inverse capabilities
emulator = PolyEmu(X, Y, forward=True, backward=True)

# Forward prediction: parameters → observables
X_new = ...       # New parameter samples, shape (k, n)
Y_pred = emulator.forward_emulator(X_new)

# Inverse estimation: observables → parameters  
Y_new = ...       # New observable samples, shape (k, m)
X_est = emulator.backward_emulator(Y_new)

# Get symbolic polynomial expressions
forward_expressions = emulator.generate_forward_symb_emu()
backward_expressions = emulator.generate_backward_symb_emu()
```

## 📚 Examples & Applications

For detailed examples and real-world applications, including:
- **PolyCAMB‑Dℓ**: Cosmological parameter → CMB power spectrum emulation
- **PolyCAMB‑peak**: Bidirectional parameter ↔ acoustic peak mapping
- Complete Jupyter notebooks with step-by-step tutorials

Visit the examples repository: **[MomentEmu-PolyCAMB-examples](https://github.com/zzhang0123/MomentEmu-PolyCAMB-examples)**

---

### 🧠 How It Works

MomentEmu builds:

- A **moment matrix**  
  $M_{\alpha\beta} = \frac{1}{N} \sum_{i} \theta_i^\alpha \theta_i^\beta$

- A **moment vector**  
  $\nu_\alpha = \frac{1}{N} \sum_{i} \theta_i^\alpha y_i$

Solving $M c = \nu$ finds polynomial coefficients $c$. No iterative optimization is needed --- model selection uses validation RMSE.  
[Read more in the arXiv paper](https://arxiv.org/abs/2507.02179). 


