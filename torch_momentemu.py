"""
PyTorch-based auto-differentiable MomentEmu implementation

This module provides PyTorch integration for MomentEmu, enabling:
- Native PyTorch nn.Module integration
- Automatic gradient computation via autograd
- GPU acceleration with CUDA support
- Seamless ML pipeline integration

Key components:
- TorchMomentEmu: PyTorch nn.Module wrapper for MomentEmu
- create_torch_emulator(): Convert trained MomentEmu to PyTorch format
- demo_torch_autodiff(): Demonstration of PyTorch auto-differentiation capabilities
"""

import torch
import torch.nn as nn
import numpy as np
from MomentEmu import PolyEmu

class TorchMomentEmu(nn.Module):
    """PyTorch module for differentiable MomentEmu."""
    
    def __init__(self, trained_emulator):
        super().__init__()
        
        # Convert coefficients to PyTorch parameters
        self.coeffs = nn.Parameter(
            torch.tensor(trained_emulator.forward_coeffs, dtype=torch.float32),
            requires_grad=False  # Coefficients are fixed after training
        )
        
        # Store multi-indices and scaling parameters
        self.multi_indices = trained_emulator.forward_multi_indices
        self.register_buffer('input_mean', 
                           torch.tensor(trained_emulator.scaler_X.mean_, dtype=torch.float32))
        self.register_buffer('input_scale', 
                           torch.tensor(trained_emulator.scaler_X.scale_, dtype=torch.float32))
        self.register_buffer('output_mean', 
                           torch.tensor(trained_emulator.scaler_Y.mean_, dtype=torch.float32))
        self.register_buffer('output_scale', 
                           torch.tensor(trained_emulator.scaler_Y.scale_, dtype=torch.float32))
    
    def evaluate_monomials(self, X_scaled):
        """Evaluate monomials using PyTorch operations."""
        if X_scaled.dim() == 1:
            X_scaled = X_scaled.unsqueeze(0)
        
        batch_size, n_features = X_scaled.shape
        n_terms = len(self.multi_indices)
        
        # Initialize Phi matrix
        Phi = torch.ones(batch_size, n_terms, device=X_scaled.device, dtype=X_scaled.dtype)
        
        for j, alpha in enumerate(self.multi_indices):
            monomial = torch.ones(batch_size, device=X_scaled.device, dtype=X_scaled.dtype)
            for i, deg in enumerate(alpha):
                if deg > 0:
                    monomial = monomial * (X_scaled[:, i] ** deg)
            Phi[:, j] = monomial
        
        return Phi
    
    def forward(self, X):
        """Forward pass through the emulator."""
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        # Scale inputs
        X_scaled = (X - self.input_mean) / self.input_scale
        
        # Evaluate polynomials
        Phi = self.evaluate_monomials(X_scaled)
        
        # Predict scaled outputs
        Y_scaled = Phi @ self.coeffs
        
        # Unscale outputs
        Y = Y_scaled * self.output_scale + self.output_mean
        
        return Y

# Alias for the class name expected by the test
MomentEmuModule = TorchMomentEmu

def create_torch_emulator(trained_emulator):
    """Create a PyTorch emulator from a trained MomentEmu."""
    return TorchMomentEmu(trained_emulator)

def demo_torch_autodiff():
    """Demonstrate auto-differentiation with PyTorch."""
    
    # Train regular MomentEmu
    print("Training MomentEmu...")
    np.random.seed(42)
    X_train = np.random.uniform(-1, 1, (100, 3))
    Y_train = (X_train[:, 0]**2 + X_train[:, 1]*X_train[:, 2]).reshape(-1, 1)
    
    emulator = PolyEmu(X_train, Y_train, forward=True, backward=False)
    
    # Convert to PyTorch
    print("Converting to PyTorch...")
    torch_emu = TorchMomentEmu(emulator)
    
    # Test point (requires gradient)
    x_test = torch.tensor([0.5, 0.3, 0.2], requires_grad=True, dtype=torch.float32)
    
    # Forward pass
    y_pred = torch_emu(x_test)
    print(f"Prediction: {y_pred.item()}")
    
    # Compute gradient via backpropagation
    y_pred.backward()
    print(f"Gradient: {x_test.grad}")
    
    # Compute Jacobian for multiple outputs (if needed)
    x_test2 = torch.tensor([0.5, 0.3, 0.2], requires_grad=True, dtype=torch.float32)
    y_pred2 = torch_emu(x_test2)
    
    # Use autograd to compute Jacobian
    jacobian = torch.autograd.functional.jacobian(torch_emu, x_test2)
    print(f"Jacobian shape: {jacobian.shape}")
    print(f"Jacobian: {jacobian}")

if __name__ == "__main__":
    demo_torch_autodiff()
