# MomentEmu Documentation

Welcome to the official documentation for **MomentEmu**, a lightweight, interpretable polynomial emulator for smooth mappings with auto-differentiation support.

## What is MomentEmu?

MomentEmu implements the **moment-projection polynomial emulator** introduced in Zhang (2025) ([arXiv:2507.02179](https://arxiv.org/abs/2507.02179)). It builds interpretable, closed-form polynomial emulators via moment matrices, achieving millisecond-level inference and symbolic transparency.

## Key Features

- ✨ **Pure Python implementation** with minimal dependencies
- 🚀 **Fast training** via moment matrices; near-instant inference
- 🔄 **Bidirectional emulation**: forward (θ → y) and inverse (y → θ)
- 🧮 **Auto-differentiation support** via JAX, PyTorch, and SymPy
- 📊 **Symbolic expressions** for full interpretability
- 🎯 **Suitable for** MCMC, Bayesian inference, sensitivity analyses
- 📦 **Compact** - no heavy model files

## Quick Start

```python
from MomentEmu import PolyEmu

# Create emulator
emulator = PolyEmu(X_train, Y_train, forward=True, backward=True)

# Forward prediction: parameters → observables
Y_pred = emulator.forward_emulator(X_new)

# Inverse estimation: observables → parameters
X_est = emulator.backward_emulator(Y_new)

# Get symbolic expressions
forward_expr = emulator.generate_forward_symb_emu()
```

## Auto-Differentiation Frameworks

MomentEmu supports three auto-differentiation frameworks:

=== "JAX"
    High-performance computing with JIT compilation
    ```python
    from jax_momentemu import create_jax_emulator
    jax_emu = create_jax_emulator(emulator)
    ```

=== "PyTorch"
    Neural network integration and ML pipelines
    ```python
    from torch_momentemu import create_torch_emulator
    torch_emu = create_torch_emulator(emulator)
    ```

=== "SymPy"
    Exact symbolic differentiation
    ```python
    from symbolic_momentemu import create_symbolic_emulator
    sym_emu = create_symbolic_emulator(emulator)
    ```

## Navigation

Use the navigation menu to explore:

- **[Installation](installation/quick-start.md)** - Get started with MomentEmu
- **[Tutorials](tutorials/getting-started.md)** - Step-by-step guides
- **[API Reference](api/core.md)** - Detailed function documentation
- **[Theory](theory/mathematical-background.md)** - Mathematical foundations
- **[Examples](examples/use-cases.md)** - Real-world applications

## External Resources

- 📄 **Paper**: [arXiv:2507.02179](https://arxiv.org/abs/2507.02179)
- 🧪 **Examples**: [MomentEmu-PolyCAMB-examples](https://github.com/MomentEmu/MomentEmu-PolyCAMB-examples)
- 🐛 **Issues**: [GitHub Issues](https://github.com/zzhang0123/MomentEmu/issues)

## How It Works

MomentEmu builds polynomial emulators by solving a linear system:

- **Moment matrix**: $M_{\alpha\beta} = \frac{1}{N} \sum_{i} \theta_i^\alpha \theta_i^\beta$
- **Moment vector**: $\nu_\alpha = \frac{1}{N} \sum_{i} \theta_i^\alpha y_i$
- **Solution**: $M c = \nu$ finds polynomial coefficients $c$

No iterative optimization needed! Model selection uses validation RMSE.
