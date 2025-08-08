# Use Cases and Applications

Explore real-world applications where MomentEmu excels.

## Scientific Computing

### Cosmological Parameter Estimation

**PolyCAMB Example**: Emulating the relationship between cosmological parameters and CMB power spectra.

```python
# Example: 6-parameter ΛCDM model → CMB power spectrum
cosmological_params = ['Ωₘ', 'Ωᵦ', 'h', 'nₛ', 'ln(10¹⁰Aₛ)', 'τ']
cmb_spectrum = f_CAMB(cosmological_params)  # Expensive CAMB computation

# Train bidirectional emulator
emulator = PolyEmu(cosmological_params, cmb_spectrum, 
                   forward=True, backward=True)

# Fast parameter estimation from observations
observed_spectrum = load_planck_data()
estimated_params = emulator.backward_emulator(observed_spectrum)
```

**Benefits**:
- ⚡ 1000× faster than CAMB
- 🎯 Maintains cosmological accuracy
- 🔄 Bidirectional parameter-observable mapping

### Physics Simulations

**Accelerate expensive simulations** in particle physics, fluid dynamics, climate modeling:

```python
# Replace expensive PDE solver with polynomial emulator
simulation_params = get_simulation_grid()  # Temperature, pressure, etc.
simulation_results = run_expensive_pde_solver(simulation_params)

emulator = PolyEmu(simulation_params, simulation_results, forward=True)

# Real-time parameter sweeps
new_params = optimize_experimental_conditions()
predicted_results = emulator.forward_emulator(new_params)
```

## Machine Learning Integration

### Neural Network Components

**Hybrid architectures** combining interpretable polynomials with neural networks:

```python
import torch.nn as nn
from MomentEmu.torch_momentemu import create_torch_emulator

class HybridPhysicsNN(nn.Module):
    def __init__(self, physics_emulator):
        super().__init__()
        # Physics-informed component (interpretable)
        self.physics = create_torch_emulator(physics_emulator)
        # Data-driven component (flexible)
        self.neural_net = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
    def forward(self, x):
        physics_pred = self.physics(x)
        residual = self.neural_net(x)
        return physics_pred + residual
```

### Transfer Learning

**Pre-trained emulators** as starting points for new problems:

```python
# Pre-trained on large simulation dataset
base_emulator = PolyEmu(large_dataset_X, large_dataset_Y, forward=True)

# Fine-tune on smaller, specific dataset
coeffs_base = base_emulator.coeffs_forward
# Use as initialization for new problem...
```

## Optimization and Inference

### Bayesian Parameter Estimation

**MCMC sampling** with fast likelihood evaluation:

```python
import emcee

def log_likelihood(params):
    predicted_obs = emulator.forward_emulator(params.reshape(1, -1))
    return -0.5 * np.sum((predicted_obs - observed_data)**2 / sigma**2)

# Fast MCMC sampling
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood)
sampler.run_mcmc(initial_guess, nsteps)
```

### Gradient-Based Optimization

**Auto-differentiation** for efficient optimization:

```python
# JAX optimization
from jax import grad
from jax.scipy.optimize import minimize

jax_emu = create_jax_emulator(emulator)
objective = lambda x: jnp.sum((jax_emu(x) - target)**2)
gradient_fn = grad(objective)

result = minimize(objective, x0, jac=gradient_fn, method='BFGS')
```

## Sensitivity Analysis

### Parameter Importance

**Analyze parameter sensitivity** using symbolic derivatives:

```python
# Get symbolic expressions
symbolic_emu = create_symbolic_emulator(emulator, ['x1', 'x2', 'x3'])
expressions = symbolic_emu['expression']
gradients = symbolic_emu['gradient']

# Evaluate sensitivity at different points
sensitivity_map = {}
for param_name, grad_expr in zip(['x1', 'x2', 'x3'], gradients):
    sensitivity_map[param_name] = grad_expr.subs([(x1, 0.5), (x2, 0.3), (x3, 0.7)])
```

### Uncertainty Propagation

**Propagate parameter uncertainties** through the emulator:

```python
# Monte Carlo uncertainty propagation
param_samples = np.random.multivariate_normal(mean_params, cov_params, 10000)
output_samples = emulator.forward_emulator(param_samples)
output_uncertainty = np.std(output_samples, axis=0)
```

## Real-Time Applications

### Control Systems

**Real-time system control** with fast emulator responses:

```python
class RealTimeController:
    def __init__(self, system_emulator):
        self.emulator = system_emulator
        
    def control_update(self, current_state, target_state):
        # Predict system response to different control inputs
        control_candidates = generate_control_grid()
        predicted_states = self.emulator.forward_emulator(control_candidates)
        
        # Choose best control action
        best_idx = np.argmin(np.linalg.norm(predicted_states - target_state, axis=1))
        return control_candidates[best_idx]
```

### Interactive Simulations

**Real-time parameter exploration** in scientific software:

```python
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def interactive_explorer(emulator):
    fig, ax = plt.subplots()
    
    def update_plot(val):
        params = [slider.val for slider in sliders]
        result = emulator.forward_emulator(np.array(params).reshape(1, -1))
        line.set_ydata(result.flatten())
        fig.canvas.draw()
    
    # Create sliders for each parameter
    sliders = [Slider(ax_slider, f'Param {i}', 0, 1, valinit=0.5) 
               for i, ax_slider in enumerate(slider_axes)]
    
    for slider in sliders:
        slider.on_changed(update_plot)
```

## Performance Benchmarks

### Speed Comparisons

| Method | Training Time | Inference Time | Use Case |
|--------|---------------|----------------|----------|
| **MomentEmu** | 0.1s | 0.001s | Fast approximation |
| **Gaussian Process** | 10s | 0.1s | Uncertainty quantification |
| **Neural Network** | 100s | 0.01s | Complex patterns |
| **Full Simulation** | 1000s | 1000s | Ground truth |

### Accuracy Benchmarks

For typical scientific applications:
- **Forward emulation**: 0.1-1% relative error
- **Inverse emulation**: 1-5% parameter recovery error
- **Gradient accuracy**: Near-exact (symbolic derivatives)

## Best Practices

### When to Use MomentEmu

✅ **Ideal for**:
- Smooth, continuous mappings
- Need for interpretability
- Fast inference requirements
- Limited training data
- Gradient-based optimization

❌ **Not suitable for**:
- Highly discontinuous functions
- Very high-dimensional outputs (>100)
- Categorical/discrete outputs
- Deep learning feature extraction

### Parameter Selection

- **Polynomial degree**: Start low (3-5), increase if needed
- **Training data**: 10-50× more samples than polynomial terms
- **Input scaling**: Always standardize inputs
- **Validation**: Use held-out data for model selection

## Next Steps

- Explore [Performance Benchmarks](benchmarks.md) for detailed comparisons
- Check [External Examples](polycamb.md) for PolyCAMB integration
- See [Tutorials](../tutorials/getting-started.md) for step-by-step guides
