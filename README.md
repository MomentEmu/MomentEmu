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
- **Modular auto-differentiation support** via JAX, PyTorch, and SymPy
- Suitable for MCMC, Bayesian inference, sensitivity analyses, and gradient-based tasks
- Compact—no heavy model files
- **Flexible installation** with optional dependencies for different use cases

## 🛠️ Installation

### Basic Installation
```bash
# Core functionality only (lightweight)
pip install git+https://github.com/zzhang0123/MomentEmu.git
```

### With Auto-Differentiation Support
```bash
# Core + JAX (high-performance computing)
pip install "git+https://github.com/zzhang0123/MomentEmu.git[jax]"

# Core + PyTorch (machine learning)
pip install "git+https://github.com/zzhang0123/MomentEmu.git[torch]"

# Core + all auto-differentiation frameworks
pip install "git+https://github.com/zzhang0123/MomentEmu.git[autodiff]"

# Everything including visualization tools
pip install "git+https://github.com/zzhang0123/MomentEmu.git[all]"
```

### Development Installation
```bash
git clone https://github.com/zzhang0123/MomentEmu.git
cd MomentEmu
pip install -e .[all]  # Install in development mode with all features
```

## 📋 Dependencies

### Core Dependencies (always installed)
- `numpy`
- `scipy` 
- `sympy`
- `scikit-learn`

### Optional Dependencies (install as needed)
- **JAX**: `jax`, `jaxlib` (for high-performance auto-differentiation)
- **PyTorch**: `torch` (for machine learning integration)  
- **Visualization**: `matplotlib` (for plotting and analysis)

**📋 Package Configuration Details**: See [PACKAGE_CONFIG_SUMMARY.md](PACKAGE_CONFIG_SUMMARY.md) for complete information about the package structure, installation options, and auto-differentiation module integration.

## 🧪 Quick Start

**Note**: Make sure to install MomentEmu first using one of the installation methods above.

```python
from MomentEmu import PolyEmu

# Define your training data
X = ...  # Input parameters, shape (N, n)
Y = ...  # Output observables, shape (N, m)

# Create emulator with both forward and inverse capabilities
emulator = PolyEmu(X, Y, 
                   forward=True,              # Enable forward emulation: parameters → observables
                   max_degree_forward=10,     # Max polynomial degree for forward mapping (lower for high-dimensional problems)
                   backward=True,             # Enable inverse emulation: observables → parameters (requires non-degenerate mapping)
                   max_degree_backward=3,     # Max polynomial degree for inverse mapping (lower for high-dimensional problems)
                   )

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

## Auto-Differentiation Support

**MomentEmu supports automatic differentiation** through three different frameworks, enabling gradient-based optimization, neural network integration, and exact symbolic analysis:

### Available Frameworks:
- **🚀 JAX**: High-performance computing with JIT compilation and GPU acceleration
- **🔥 PyTorch**: Native neural network integration and ML pipeline compatibility  
- **🔢 SymPy**: Exact symbolic differentiation with zero numerical error

### Quick Example:
```python
# JAX implementation
from jax_momentemu import create_jax_emulator
import jax.numpy as jnp
from jax import grad

# Convert trained emulator to JAX
jax_emu = create_jax_emulator(emulator)

# Compute gradients automatically
x = jnp.array([0.5, 0.3])
y = jax_emu(x)
gradient = grad(lambda x: jax_emu(x).sum())(x)
```

### 📖 Complete Auto-Differentiation Guide
For comprehensive documentation, performance comparisons, usage examples, and integration guidelines, see:

**[📋 AUTODIFF_GUIDE.md](AUTODIFF_GUIDE.md)**

The guide covers:
- Detailed usage for each framework (JAX, PyTorch, SymPy)
- Performance benchmarks and framework comparison
- Integration guidelines for different use cases
- Complete testing suite and troubleshooting tips

**[⚙️ Package Configuration Details](PACKAGE_CONFIG_SUMMARY.md)** - Complete information about package structure, installation options, and module integration.

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


---
---

### Appendix: Derivative Errors in Polynomial Approximations

When a smooth function $f$ is approximated by a polynomial $P_n$ of degree $n$, the error in the approximation of its derivatives generally amplifies with the derivative order. If the function is fitted to a uniform accuracy $\|f - P_n\|_\infty \leq \delta $, then the worst-case error in the $r$-th derivative satisfies the bound
$$
\|f^{(r)} - P_n^{(r)}\|_\infty \lesssim n^r \cdot \delta,
$$
reflecting the fact that differentiation acts as a numerically unstable operator in the space of polynomials. This growth arises from Bernstein-type inequalities and classical results in approximation theory, such as Jackson’s theorem. Thus, while polynomial emulation can be highly accurate for the function itself, care must be taken when using it to infer high-order derivatives, especially for large $n$ or high $r$, as derivative estimates can become significantly less accurate even when the original approximation error is small.

---

### 📚 References

1. **Timothy J. Rivlin**, *An Introduction to the Approximation of Functions*  
   – Classic and accessible introduction. See Chapter 4–5 on uniform polynomial approximation and error estimates.

2. **E. W. Cheney**, *Introduction to Approximation Theory*  
   – Comprehensive, rigorous treatment. Bernstein and Jackson inequalities are covered in detail.

3. **L. N. Trefethen**, *Spectral Methods in MATLAB*  
   – Discusses how polynomial interpolation and spectral approximations behave under differentiation; very readable with practical insights.


