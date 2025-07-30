"""
JAX-based auto-differentiable MomentEmu implementation

This module provides JAX integration for MomentEmu, enabling:
- High-performance computing with JIT compilation
- Automatic differentiation for gradients, Jacobians, and Hessians
- GPU acceleration support
- Vectorized batch operations

Key functions:
- create_jax_emulator(): Convert trained MomentEmu to JAX format
- demo_jax_autodiff(): Demonstration of JAX auto-differentiation capabilities
"""

import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev
import numpy as np
from MomentEmu import PolyEmu

def create_jax_emulator(emulator):
    """Convert trained MomentEmu to JAX-differentiable function."""
    
    # Extract learned parameters
    coeffs = jnp.array(emulator.forward_coeffs)
    multi_indices = emulator.forward_multi_indices
    
    # Extract scaling parameters
    input_mean = jnp.array(emulator.scaler_X.mean_)
    input_scale = jnp.array(emulator.scaler_X.scale_)
    output_mean = jnp.array(emulator.scaler_Y.mean_)
    output_scale = jnp.array(emulator.scaler_Y.scale_)
    
    def evaluate_monomials_jax(X_scaled, multi_indices):
        """JAX-compatible monomial evaluation."""
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(1, -1)
        
        N, n = X_scaled.shape
        D = len(multi_indices)
        
        Phi = jnp.ones((N, D))
        for j, alpha in enumerate(multi_indices):
            monomial = jnp.ones(N)
            for i, deg in enumerate(alpha):
                if deg > 0:
                    monomial = monomial * (X_scaled[:, i] ** deg)
            Phi = Phi.at[:, j].set(monomial)
        
        return Phi
    
    @jax.jit
    def jax_emulator(X):
        """JAX-compiled differentiable emulator."""
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Scale inputs
        X_scaled = (X - input_mean) / input_scale
        
        # Evaluate polynomials
        Phi = evaluate_monomials_jax(X_scaled, multi_indices)
        
        # Predict scaled outputs
        Y_scaled = Phi @ coeffs
        
        # Unscale outputs
        Y = Y_scaled * output_scale + output_mean
        
        return Y
    
    return jax_emulator

# Example usage
def demo_jax_autodiff():
    """Demonstrate auto-differentiation with JAX."""
    
    # Train regular MomentEmu
    print("Training MomentEmu...")
    np.random.seed(42)
    X_train = np.random.uniform(-1, 1, (100, 2))
    Y_train = (X_train[:, 0]**2 + X_train[:, 1]**2).reshape(-1, 1)
    
    emulator = PolyEmu(X_train, Y_train, forward=True, backward=False)
    
    # Convert to JAX
    print("Converting to JAX...")
    jax_emu = create_jax_emulator(emulator)
    
    # Test point
    x_test = jnp.array([0.5, 0.3])
    
    # Forward pass
    y_pred = jax_emu(x_test)
    print(f"Prediction: {y_pred}")
    
    # Compute gradient
    grad_fn = grad(lambda x: jax_emu(x).sum())
    gradient = grad_fn(x_test)
    print(f"Gradient: {gradient}")
    
    # Compute Jacobian
    jac_fn = jacfwd(jax_emu)
    jacobian = jac_fn(x_test)
    print(f"Jacobian shape: {jacobian.shape}")
    print(f"Jacobian: {jacobian}")
    
    # Compute Hessian
    hess_fn = jacfwd(jacrev(lambda x: jax_emu(x).sum()))
    hessian = hess_fn(x_test)
    print(f"Hessian: {hessian}")

if __name__ == "__main__":
    demo_jax_autodiff()
