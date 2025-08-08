# Quick Start Installation

Get up and running with MomentEmu in minutes.

## System Requirements

- Python 3.7 or higher
- NumPy, SciPy, SymPy, scikit-learn (automatically installed)

## Basic Installation

For core functionality only:

```bash
pip install git+https://github.com/zzhang0123/MomentEmu.git
```

This installs the lightweight version with all essential features for polynomial emulation.

## Installation with Auto-Differentiation

Choose your preferred framework:

=== "JAX (Recommended for HPC)"
    ```bash
    pip install "git+https://github.com/zzhang0123/MomentEmu.git[jax]"
    ```
    Best for high-performance computing, scientific research, and GPU acceleration.

=== "PyTorch (ML Integration)"
    ```bash
    pip install "git+https://github.com/zzhang0123/MomentEmu.git[torch]"
    ```
    Perfect for machine learning pipelines and neural network integration.

=== "All Frameworks"
    ```bash
    pip install "git+https://github.com/zzhang0123/MomentEmu.git[autodiff]"
    ```
    Installs both JAX and PyTorch support.

=== "Everything"
    ```bash
    pip install "git+https://github.com/zzhang0123/MomentEmu.git[all]"
    ```
    Includes visualization tools (matplotlib) and all features.

## Verify Installation

Test your installation:

```python
import numpy as np
from MomentEmu import PolyEmu

# Generate test data
X = np.random.rand(100, 2)
Y = (X[:, 0]**2 + X[:, 1]**2).reshape(-1, 1)

# Create emulator
emulator = PolyEmu(X, Y, forward=True)
print("✅ MomentEmu installed successfully!")

# Test auto-differentiation (if installed)
try:
    from MomentEmu import jax_momentemu
    print("✅ JAX support available")
except ImportError:
    print("⚠️ JAX not installed")

try:
    from MomentEmu import torch_momentemu
    print("✅ PyTorch support available")
except ImportError:
    print("⚠️ PyTorch not installed")
```

## Next Steps

- Continue to [Getting Started Tutorial](../tutorials/getting-started.md)
- Learn about [Auto-Differentiation Setup](autodiff.md)
- Explore [Development Installation](development.md) for contributors

## Troubleshooting

!!! warning "Common Issues"
    - **Import errors**: Make sure you're using Python 3.7+
    - **JAX installation**: On some systems, you may need platform-specific JAX installation
    - **PyTorch installation**: Visit [pytorch.org](https://pytorch.org) for platform-specific instructions

!!! tip "Performance Tips"
    - Use JAX installation for fastest performance on large datasets
    - The basic installation is sufficient for most use cases
    - Consider GPU-enabled JAX for very large problems
