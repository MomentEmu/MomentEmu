import numpy as np
from itertools import combinations_with_replacement
from collections import Counter
import sympy as sp
from logging import warning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed



def generate_multi_indices(n, d):
    """Generate all multi-indices α in ℕ^n with total degree ≤ d."""
    indices = []
    for deg in range(d + 1):
        for c in combinations_with_replacement(range(n), deg):
            counter = Counter(c)
            alpha = [counter[i] for i in range(n)]
            indices.append(tuple(alpha))
    return indices

# def evaluate_monomials(X, multi_indices, n_jobs=-1):
#     """
#     Evaluate monomials φ_α(X) = X^α for each sample and each multi-index α, in parallel.

#     Parameters:
#         X : array, shape (N, n)
#             Input samples
#         multi_indices : list of tuple, shape (D,)
#             Multi-indices α
#         n_jobs : int
#             Number of parallel jobs (-1 = all cores)

#     Returns:
#         Phi : array, shape (N, D)
#             Evaluated monomials
#     """
#     def _eval_alpha(alpha):
#         return np.prod(X ** alpha, axis=1)
    
#     Phi_cols = Parallel(n_jobs=n_jobs, backend='threading')(
#         delayed(_eval_alpha)(alpha) for alpha in multi_indices
#     )

#     return np.column_stack(Phi_cols)

# def compute_moments_vector_output(X, Y, degree, n_jobs=-1):
#     """
#     Vector-valued moment method using joblib-parallelized monomial evaluation.

#     Parameters:
#         X : array, shape (N, n)
#         Y : array, shape (N, m)
#         degree : int, maximum total polynomial degree
#         n_jobs : int, number of parallel jobs for monomial evaluation

#     Returns:
#         M : array, shape (D, D)
#             Moment matrix (Gram matrix)
#         nu : array, shape (D, m)
#             Projected observable moments
#         multi_indices : list of tuple
#             All multi-indices α used
#     """
#     N, n = X.shape

#     multi_indices = generate_multi_indices(n, degree)
    
#     Phi = evaluate_monomials(X, multi_indices, n_jobs=n_jobs)  # N x D (D = len(multi_indices))

#     M = (Phi.T @ Phi) / N     # D x D
#     nu = (Phi.T @ Y) / N      # D x m (m = Y.shape[1])

#     return M, nu, multi_indices


def evaluate_monomials(X, multi_indices):
    """Evaluate φ_α(X) for all samples and all α."""
    N, n = X.shape
    D = len(multi_indices)
    Phi = np.empty((N, D))
    for j, alpha in enumerate(multi_indices):
        Phi[:, j] = np.prod(X ** alpha, axis=1)
    return Phi  # shape: N x D

def evaluate_monomials_lazy(X, multi_indices):
    """
    Efficiently evaluate monomials using on-demand caching to reduce memory use.
    """
    N, n = X.shape
    D = len(multi_indices)
    
    # Cache only needed powers: (i, d) -> X[:, i] ** d
    power_cache = {}
    
    Phi = np.empty((N, D))
    for j, alpha in enumerate(multi_indices):
        phi_j = np.ones(N)
        for i, deg in enumerate(alpha):
            if deg == 0:
                continue
            key = (i, deg)
            if key not in power_cache:
                power_cache[key] = X[:, i] ** deg
            phi_j *= power_cache[key]
        Phi[:, j] = phi_j
    return Phi

def compute_moments_vector_output(X, Y, degree):
    """
    Vector-valued version of moment method.
    X: N x n input parameter array
    Y: N x m observable array
    degree: total degree of monomials
    Returns: moment matrix M, moment vectors ν (D x m), and multi-indices
    """
    N, n = X.shape
    m = Y.shape[1]  # number of outputs

    multi_indices = generate_multi_indices(n, degree)
    D = len(multi_indices)
    Phi = evaluate_monomials_lazy(X, multi_indices)  # N x D

    M = (Phi.T @ Phi) / N                       # D x D
    nu = (Phi.T @ Y) / N                        # D x m

    return M, nu, multi_indices

def solve_emulator_coefficients(M, nu):
    """
    Solve Mc = ν for each output dimension
    Returns: coefficients array of shape D x m
    """
    return np.linalg.solve(M, nu)  # D x m

def symbolic_polynomial_expressions(coeffs, multi_indices, variable_names=None):
    """
    Convert emulator coefficients into sympy expressions.
    coeffs: D x m (number of basis terms × number of outputs)
    multi_indices: list of α
    Returns: list of sympy expressions, one per output dimension
    """
    D, m = coeffs.shape
    n = len(multi_indices[0])
    if variable_names is None:
        variable_names = [f"x{i+1}" for i in range(n)]
    vars_sym = sp.symbols(variable_names)

    expressions = []
    for j in range(m):  # For each output dimension
        expr = 0
        for c, alpha in zip(coeffs[:, j], multi_indices):
            monomial = np.prod([vars_sym[i]**alpha[i] for i in range(n)])
            expr += c * monomial
        expressions.append(sp.simplify(expr))
    return expressions, vars_sym

def evaluate_emulator(X, coeffs, multi_indices):
    """
    Evaluate the polynomial emulator at inputs X using known coefficients.
    X: N x n
    coeffs: D x m
    multi_indices: list of α
    Returns: Y_pred: N x m
    """
    Phi = evaluate_monomials_lazy(X, multi_indices)  # N x D
    return Phi @ coeffs  # N x m


class PolyEmu():
    def __init__(self, X, Y, X_test=None, Y_test=None, test_size=0.2, 
                RMSE_tol=1e-2, forward=True, backward=False,
                init_deg_forward=None, max_degree_forward=10, 
                init_deg_backward=None, max_degree_backward=10, 
                fractional_error=False, 
                X_with_std=True,
                Y_with_std=True):
        self.n_params = X.shape[1]
        self.n_outputs = Y.shape[1]
        if X_test is None or Y_test is None:
            # Split into training and validation
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size)
        else:
            X_train, Y_train = X, Y
            X_val, Y_val = X_test, Y_test

        # Scale the training data
        self.scaler_X = StandardScaler(with_std=X_with_std)
        self.scaler_Y = StandardScaler(with_std=Y_with_std)

        self.X_train_scaled = self.scaler_X.fit_transform(X_train)
        self.Y_train_scaled = self.scaler_Y.fit_transform(Y_train)
        self.X_val_scaled = self.scaler_X.transform(X_val)
        self.Y_val_scaled = self.scaler_Y.transform(Y_val)

        if forward:
            print("Generating forward emulator...")
            self.forward_emulator = self.generate_forward_emulator(RMSE_tol=RMSE_tol, init_deg=init_deg_forward, max_degree=max_degree_forward, fractional_error=fractional_error)
        if backward:
            print("Generating backward emulator...")
            self.backward_emulator = self.generate_backward_emulator(RMSE_tol=RMSE_tol, init_deg=init_deg_backward, max_degree=max_degree_backward, fractional_error=fractional_error)

    def generate_forward_emulator(self, RMSE_tol=1e-3, init_deg=None, max_degree=10, fractional_error=True):

        if init_deg is None:
            if self.n_params > 6:
                degree = 1
            elif self.n_params < 3:
                degree = 3
            else:
                degree = 2
        else:
            degree = init_deg

        assert degree <= max_degree, "Initial degree must be less than or equal to max_degree"

        coeffs_list = []
        RMSE_val_list = []
        multi_indices_list = []

        for d in range(degree, max_degree + 1):
            M, nu, multi_indices = compute_moments_vector_output(self.X_train_scaled, self.Y_train_scaled, d)
            coeffs = solve_emulator_coefficients(M, nu)

            Y_val_pred = evaluate_emulator(self.X_val_scaled, coeffs, multi_indices)

            if fractional_error:
                # Define the RMS fractional error
                frac_err = (Y_val_pred-self.Y_val_scaled) / (np.abs(self.Y_val_scaled) + 1e-10)
                # Calculate the RMS fractional error
                RMSE_val = np.sqrt(np.mean(frac_err**2))
            else:
                RMSE_val = np.sqrt(mean_squared_error(self.Y_val_scaled, Y_val_pred))

            coeffs_list.append(coeffs)
            multi_indices_list.append(multi_indices)
            RMSE_val_list.append(RMSE_val)

            if RMSE_val < RMSE_tol:
                print(f"Forward emulator generated with degree {d}, RMSE_val of {RMSE_val}.")
                break
            if d == max_degree:
                # find the degree with the lowest RMSE
                ind = np.argmin(RMSE_val_list)
                coeffs = coeffs_list[ind]
                multi_indices = multi_indices_list[ind]
                warning(f"Maximum degree {max_degree} reached. Returning emulator with degree {degree+ind} with RMSE_val of {RMSE_val_list[ind]}.")
        
        def emulator(X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X_scaled = self.scaler_X.transform(X)
            Y_pred_scaled = evaluate_emulator(X_scaled, coeffs, multi_indices)
            Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)
            return Y_pred
        
        return emulator


    def generate_backward_emulator(self, RMSE_tol=1e-2, init_deg=None, max_degree=10, fractional_error=True):
        if init_deg is None:
            if self.n_outputs > 6:
                degree = 1
            elif self.n_outputs < 3:
                degree = 3
            else:
                degree = 2
        else:
            degree = init_deg

        assert degree <= max_degree, "Initial degree must be less than or equal to max_degree"

        coeffs_list = []
        RMSE_val_list = []
        multi_indices_list = []

        for d in range(degree, max_degree + 1):
            M, nu, multi_indices = compute_moments_vector_output(self.Y_train_scaled, self.X_train_scaled, d)
            coeffs = solve_emulator_coefficients(M, nu)

            X_val_pred = evaluate_emulator(self.Y_val_scaled, coeffs, multi_indices)
            RMSE_val = np.sqrt(mean_squared_error(self.X_val_scaled, X_val_pred))

            if fractional_error:
                # Define the RMS fractional error
                frac_err = (X_val_pred-self.X_val_scaled) / (np.abs(self.X_val_scaled) + 1e-10)
                # Calculate the RMS fractional error
                RMSE_val = np.sqrt(np.mean(frac_err**2))
            else:
                RMSE_val = np.sqrt(mean_squared_error(self.X_val_scaled, X_val_pred))

            coeffs_list.append(coeffs)
            multi_indices_list.append(multi_indices)
            RMSE_val_list.append(RMSE_val)

            if RMSE_val < RMSE_tol:
                print(f"Backward emulator generated with degree {d}, RMSE_val of {RMSE_val}.")
                break
            if d == max_degree:
                # find the degree with the lowest RMSE
                ind = np.argmin(RMSE_val_list)
                coeffs = coeffs_list[ind]
                multi_indices = multi_indices_list[ind]
                warning(f"Maximum degree {max_degree} reached. Returning emulator with degree {degree+ind} with RMSE_val of {RMSE_val_list[ind]}.")

        def emulator(Y):
            Y_scaled = self.scaler_Y.transform(Y)
            X_pred_scaled = evaluate_emulator(Y_scaled, coeffs, multi_indices)
            X_pred = self.scaler_X.inverse_transform(X_pred_scaled)
            return X_pred

        return emulator
