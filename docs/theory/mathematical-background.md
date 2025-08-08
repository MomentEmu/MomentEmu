# Mathematical Background

Understanding the mathematical foundations of MomentEmu's polynomial emulation approach.

## Core Concept

MomentEmu builds polynomial emulators by solving a **linear system** derived from moment matrices, avoiding iterative optimization entirely.

## The Moment-Projection Method

### Problem Setup

Given training data $\{(\theta_i, y_i)\}_{i=1}^N$ where:
- $\theta_i \in \mathbb{R}^n$ are input parameters
- $y_i \in \mathbb{R}^m$ are output observables

We want to find a polynomial mapping $P: \mathbb{R}^n \to \mathbb{R}^m$ such that:
$$P(\theta_i) \approx y_i \quad \text{for all } i$$

### Polynomial Basis

For a given maximum degree $d$, we construct a polynomial basis using **multi-indices**:
$$\{\theta^{\alpha} : \alpha \in \mathcal{A}_d\}$$

where $\mathcal{A}_d = \{\alpha \in \mathbb{N}_0^n : |\alpha| \leq d\}$ and $\theta^{\alpha} = \prod_{j=1}^n \theta_j^{\alpha_j}$.

!!! example "Example: 2D quadratic basis"
    For $n=2, d=2$: $\{1, \theta_1, \theta_2, \theta_1^2, \theta_1\theta_2, \theta_2^2\}$

### Moment Matrix Construction

The **moment matrix** $M \in \mathbb{R}^{|\mathcal{A}_d| \times |\mathcal{A}_d|}$ is defined as:
$$M_{\alpha\beta} = \frac{1}{N} \sum_{i=1}^N \theta_i^{\alpha} \theta_i^{\beta}$$

This captures the correlations between all polynomial basis functions.

### Moment Vector Construction

For each output dimension $j$, the **moment vector** $\nu^{(j)} \in \mathbb{R}^{|\mathcal{A}_d|}$ is:
$$\nu^{(j)}_{\alpha} = \frac{1}{N} \sum_{i=1}^N \theta_i^{\alpha} y_i^{(j)}$$

This represents the correlation between basis functions and output values.

### Linear System Solution

The polynomial coefficients $c^{(j)}$ for output dimension $j$ are found by solving:
$$M c^{(j)} = \nu^{(j)}$$

This is a **linear system** - no iterative optimization required!

### Final Polynomial

The resulting polynomial emulator is:
$$P^{(j)}(\theta) = \sum_{\alpha \in \mathcal{A}_d} c^{(j)}_{\alpha} \theta^{\alpha}$$

## Key Advantages

### 1. Non-Iterative Training
- **Traditional ML**: Requires gradient descent, backpropagation, hyperparameter tuning
- **MomentEmu**: Direct linear algebra solution

### 2. Interpretable Results
- Coefficients have clear mathematical meaning
- Symbolic expressions available
- Easy to analyze polynomial structure

### 3. Fast Inference
- Simple polynomial evaluation
- No complex model architecture
- Vectorizable operations

### 4. Theoretical Guarantees
- Well-conditioned for appropriate data scaling
- Unique solution (when $M$ is invertible)
- Convergence properties from approximation theory

## Bidirectional Emulation

### Forward Mapping
Standard case: $\theta \to y$
$$y = P_{\text{forward}}(\theta)$$

### Inverse Mapping  
For invertible relationships: $y \to \theta$
$$\theta = P_{\text{backward}}(y)$$

MomentEmu constructs separate moment matrices for each direction:
- **Forward**: $M^{(\theta)}_{\alpha\beta} = \frac{1}{N} \sum_i \theta_i^{\alpha} \theta_i^{\beta}$
- **Backward**: $M^{(y)}_{\alpha\beta} = \frac{1}{N} \sum_i y_i^{\alpha} y_i^{\beta}$

## Computational Complexity

### Training Time
- **Moment matrix**: $O(N \cdot |\mathcal{A}_d|^2)$
- **Linear solve**: $O(|\mathcal{A}_d|^3)$
- **Total**: $O(N \cdot |\mathcal{A}_d|^2 + |\mathcal{A}_d|^3)$

where $|\mathcal{A}_d| = \binom{n+d}{d}$ (number of multi-indices).

### Inference Time
- **Polynomial evaluation**: $O(|\mathcal{A}_d|)$
- **Extremely fast** for inference

## Numerical Considerations

### Conditioning
- **Raw data**: Can lead to ill-conditioned moment matrices
- **Scaled data**: Standardization improves conditioning significantly
- **Regularization**: Ridge regression for ill-conditioned cases

### Degree Selection
- **Trade-off**: Higher degree → better fit but more coefficients
- **Rule of thumb**: $N \geq 10 \cdot |\mathcal{A}_d|$ (samples per coefficient)
- **Validation**: Use held-out data to select optimal degree

## Relationship to Other Methods

### vs. Polynomial Regression
- **Standard**: Uses least-squares fitting
- **MomentEmu**: Uses moment-based approach with better numerical properties

### vs. Gaussian Processes
- **GP**: Probabilistic, kernel-based, scales as $O(N^3)$
- **MomentEmu**: Deterministic, polynomial, scales better with data size

### vs. Neural Networks
- **NN**: Black box, requires iterative training
- **MomentEmu**: Interpretable, direct solution

## Mathematical References

1. **Approximation Theory**: Jackson's theorem, Weierstrass approximation
2. **Numerical Linear Algebra**: Matrix conditioning, regularization
3. **Polynomial Interpolation**: Vandermonde matrices, basis functions
4. **Moment Methods**: Method of moments in statistics

The mathematical foundation ensures that MomentEmu provides:
- ✅ **Theoretical soundness**
- ✅ **Computational efficiency** 
- ✅ **Interpretable results**
- ✅ **Practical applicability**
