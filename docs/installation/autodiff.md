# Auto-Differentiation Installation

Learn how to install MomentEmu with auto-differentiation support for different frameworks.

## Framework Overview

MomentEmu supports three auto-differentiation frameworks:

| Framework | Use Case | Installation Command |
|-----------|----------|---------------------|
| **JAX** | High-performance computing, research | `[jax]` |
| **PyTorch** | Machine learning, neural networks | `[torch]` |
| **SymPy** | Symbolic analysis (included in core) | Core installation |

## JAX Installation

Best for high-performance scientific computing and research applications.

### Standard Installation

```bash
pip install "git+https://github.com/zzhang0123/MomentEmu.git[jax]"
```

### Platform-Specific JAX

For optimal performance, install platform-specific JAX:

=== "CPU Only"
    ```bash
    pip install "jax[cpu]"
    pip install "git+https://github.com/zzhang0123/MomentEmu.git"
    ```

=== "CUDA (GPU)"
    ```bash
    pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install "git+https://github.com/zzhang0123/MomentEmu.git"
    ```

=== "TPU"
    ```bash
    pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install "git+https://github.com/zzhang0123/MomentEmu.git"
    ```

### Verify JAX Installation

```python
import jax
import jax.numpy as jnp
from MomentEmu import jax_momentemu

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print("✅ JAX integration ready!")
```

## PyTorch Installation

Best for machine learning applications and neural network integration.

### Standard Installation

```bash
pip install "git+https://github.com/zzhang0123/MomentEmu.git[torch]"
```

### Platform-Specific PyTorch

For GPU support, visit [pytorch.org](https://pytorch.org) for platform-specific instructions:

=== "CPU Only"
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install "git+https://github.com/zzhang0123/MomentEmu.git"
    ```

=== "CUDA 11.8"
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    pip install "git+https://github.com/zzhang0123/MomentEmu.git"
    ```

=== "CUDA 12.1"
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install "git+https://github.com/zzhang0123/MomentEmu.git"
    ```

### Verify PyTorch Installation

```python
import torch
from MomentEmu import torch_momentemu

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name()}")
print("✅ PyTorch integration ready!")
```

## All Frameworks

Install everything at once:

```bash
# All auto-differentiation frameworks
pip install "git+https://github.com/zzhang0123/MomentEmu.git[autodiff]"

# Everything including visualization tools
pip install "git+https://github.com/zzhang0123/MomentEmu.git[all]"
```

## Troubleshooting

### Common Issues

!!! warning "JAX Installation Issues"
    - **M1/M2 Mac**: Use `pip install jax[cpu]` for compatibility
    - **CUDA version mismatch**: Ensure CUDA toolkit matches JAX requirements
    - **Old jaxlib**: Update with `pip install --upgrade jax jaxlib`

!!! warning "PyTorch Installation Issues"
    - **Import errors**: Check PyTorch installation with `python -c "import torch; print(torch.__version__)"`
    - **CUDA issues**: Verify CUDA installation and version compatibility
    - **Memory errors**: Start with CPU version for testing

!!! warning "General Issues"
    - **Dependency conflicts**: Use a fresh virtual environment
    - **Import errors**: Ensure you've installed MomentEmu after the framework
    - **Version mismatches**: Use the latest versions of all packages

### Testing Your Installation

Run this comprehensive test:

```python
# Test all available frameworks
import numpy as np
from MomentEmu import PolyEmu

# Create test data
X = np.random.rand(50, 2)
Y = (X[:, 0]**2 + X[:, 1]**2).reshape(-1, 1)
emulator = PolyEmu(X, Y, forward=True)

print("Testing auto-differentiation frameworks:")

# Test JAX
try:
    from MomentEmu import jax_momentemu
    jax_emu = jax_momentemu.create_jax_emulator(emulator)
    print("✅ JAX: Working")
except ImportError:
    print("❌ JAX: Not installed")
except Exception as e:
    print(f"❌ JAX: Error - {e}")

# Test PyTorch
try:
    from MomentEmu import torch_momentemu
    torch_emu = torch_momentemu.create_torch_emulator(emulator)
    print("✅ PyTorch: Working")
except ImportError:
    print("❌ PyTorch: Not installed")
except Exception as e:
    print(f"❌ PyTorch: Error - {e}")

# Test SymPy (should always work)
try:
    from MomentEmu import symbolic_momentemu
    sym_emu = symbolic_momentemu.create_symbolic_emulator(emulator, ['x', 'y'])
    print("✅ SymPy: Working")
except Exception as e:
    print(f"❌ SymPy: Error - {e}")
```

## Next Steps

- Continue to [Getting Started Tutorial](../tutorials/getting-started.md)
- Explore [Auto-Differentiation Guide](../tutorials/autodiff-guide.md)
- See [Development Installation](development.md) for contributors
