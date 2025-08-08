# Getting Started with MomentEmu

This tutorial will walk you through the basic usage of MomentEmu for polynomial emulation.

## What You'll Learn

- How to create and train a polynomial emulator
- Forward and inverse emulation
- Evaluating emulator performance
- Getting symbolic expressions

## Basic Example

Let's start with a simple 2D function: $f(x, y) = x^2 + y^2$

```python
import numpy as np
from MomentEmu import PolyEmu
import matplotlib.pyplot as plt

# Generate training data
np.random.seed(42)
n_train = 100
X_train = np.random.uniform(-1, 1, (n_train, 2))  # Parameters
Y_train = (X_train[:, 0]**2 + X_train[:, 1]**2).reshape(-1, 1)  # Observables

print(f"Training data shape: X={X_train.shape}, Y={Y_train.shape}")
```

## Creating an Emulator

```python
# Create emulator with forward mapping
emulator = PolyEmu(
    X_train, Y_train,
    forward=True,              # Enable forward emulation: θ → y
    max_degree_forward=5,      # Maximum polynomial degree
    backward=False,            # Disable inverse for this example
    scaler_type='standardize'  # Standardize inputs for better conditioning
)

print("✅ Emulator trained successfully!")
print(f"Forward degree: {emulator.degree_forward}")
print(f"Number of coefficients: {len(emulator.coeffs_forward)}")
```

## Making Predictions

```python
# Generate test data
n_test = 20
X_test = np.random.uniform(-1, 1, (n_test, 2))
Y_true = (X_test[:, 0]**2 + X_test[:, 1]**2).reshape(-1, 1)

# Forward prediction
Y_pred = emulator.forward_emulator(X_test)

# Calculate error
error = np.abs(Y_pred - Y_true)
print(f"Mean absolute error: {np.mean(error):.6f}")
print(f"Max absolute error: {np.max(error):.6f}")
```

## Visualizing Results

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

# Plot 1: True vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(Y_true, Y_pred, alpha=0.7)
plt.plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted')
plt.grid(True, alpha=0.3)

# Plot 2: Error distribution
plt.subplot(1, 2, 2)
plt.hist(error.flatten(), bins=10, alpha=0.7, color='orange')
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Getting Symbolic Expressions

One of MomentEmu's key features is providing interpretable, symbolic polynomial expressions:

```python
# Get symbolic expressions
expressions = emulator.generate_forward_symb_emu()

print("Symbolic polynomial expression:")
print(f"f(x₀, x₁) = {expressions[0]}")

# The expression should be close to: x₀² + x₁²
```

## Bidirectional Emulation

For invertible mappings, MomentEmu can perform inverse emulation:

```python
# Create bidirectional emulator
emulator_bidir = PolyEmu(
    X_train, Y_train,
    forward=True,
    backward=True,              # Enable inverse emulation: y → θ
    max_degree_forward=5,
    max_degree_backward=3       # Usually lower degree for inverse
)

# Forward: parameters → observables
Y_forward = emulator_bidir.forward_emulator(X_test[:5])

# Inverse: observables → parameters  
X_inverse = emulator_bidir.backward_emulator(Y_forward)

print("Inverse emulation test:")
print(f"Original parameters shape: {X_test[:5].shape}")
print(f"Recovered parameters shape: {X_inverse.shape}")
print(f"Recovery error: {np.mean(np.abs(X_test[:5] - X_inverse)):.6f}")
```

## Performance Tips

!!! tip "Optimization Guidelines"
    - **Degree selection**: Start with low degrees (3-5) and increase if needed
    - **Training data**: Use 10-50× more samples than the number of polynomial terms
    - **Scaling**: Always use input scaling for better numerical conditioning
    - **Validation**: Split your data to assess generalization performance

## Next Steps

- Try [Auto-Differentiation](autodiff-guide.md) for gradient-based applications
- See [Use Cases](../examples/use-cases.md) for real-world applications
- Review [Mathematical Background](../theory/mathematical-background.md) for theoretical understanding
- Check [API Reference](../api/core.md) for complete function documentation

## Common Issues

!!! warning "Troubleshooting"
    - **Poor accuracy**: Try increasing polynomial degree or adding more training data
    - **Numerical instability**: Enable input scaling with `scaler_type='standardize'`
    - **Slow training**: Reduce polynomial degree or dataset size
    - **Inverse fails**: Check that your mapping is actually invertible
