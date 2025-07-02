import numpy as np
from itertools import combinations_with_replacement
from collections import Counter
import sympy as sp
from logging import warning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



def generate_multi_indices(n, d):
    """Generate all multi-indices α in ℕ^n with total degree ≤ d."""
    indices = []
    for deg in range(d + 1):
        for c in combinations_with_replacement(range(n), deg):
            counter = Counter(c)
            alpha = [counter[i] for i in range(n)]
            indices.append(tuple(alpha))
    return indices


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

def symbolic_polynomial_expressions(coeffs, multi_indices, variable_names=None, 
                                    input_means=None, input_vars=None, 
                                    output_means=None, output_vars=None):
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

    if input_vars is not None:
        input_stds = np.sqrt(input_vars)
    else:
        input_stds = None
    if output_vars is not None:
        output_stds = np.sqrt(output_vars)
    else:
        output_stds = None

    expressions = []
    for j in range(m):  # For each output dimension
        expr = 0
        for c, alpha in zip(coeffs[:, j], multi_indices):
            if input_means is not None and input_stds is not None:
                monomial = np.prod([ ( (vars_sym[i] - input_means[i]) / input_stds[i] )**alpha[i] for i in range(n)])
            elif input_means is not None:
                monomial = np.prod([ (vars_sym[i] - input_means[i])**alpha[i] for i in range(n)])
            expr += c * monomial
        if output_means is not None and output_stds is not None:
            expr = expr * output_stds[j] + output_means[j]
        elif output_means is not None:
            expr = expr + output_means[j]
        expressions.append(sp.simplify(expr))
    return expressions

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
    def __init__(self, 
                X, 
                Y, 
                X_test=None, 
                Y_test=None, 
                test_size=0.2, 
                RMSE_tol=1e-2, 
                forward=True, 
                backward=False,
                init_deg_forward=None, 
                max_degree_forward=10, 
                init_deg_backward=None, 
                max_degree_backward=10, 
                return_max_frac_err=True):
        self.n_params = X.shape[1]
        self.n_outputs = Y.shape[1]
        if X_test is None or Y_test is None:
            # Split into training and validation
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size)
        else:
            X_train, Y_train = X, Y
            X_val, Y_val = X_test, Y_test

        # Scale the training data
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

        X_train_scaled = self.scaler_X.fit_transform(X_train)
        Y_train_scaled = self.scaler_Y.fit_transform(Y_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        Y_val_scaled = self.scaler_Y.transform(Y_val)

        if forward:
            print("Generating forward emulator...")
            self.generate_forward_emulator(X_train_scaled, 
                                        Y_train_scaled,
                                        X_val_scaled,
                                        Y_val_scaled,
                                        RMSE_tol=RMSE_tol, 
                                        init_deg=init_deg_forward, 
                                        max_degree=max_degree_forward)
            if return_max_frac_err:
                Y_val_pred = self.forward_emulator(X_val)
                max_frac_err = np.max(np.abs((Y_val_pred - Y_val) / (Y_val+1e-10)))
                self.forward_max_frac_err = max_frac_err
                print(f"Forward emulator maximum fractional error: {max_frac_err}")

        if backward:
            print("Generating backward emulator...")
            self.generate_backward_emulator(X_train_scaled, 
                                            Y_train_scaled,
                                            X_val_scaled,
                                            Y_val_scaled,
                                            RMSE_tol=RMSE_tol, 
                                            init_deg=init_deg_backward, 
                                            max_degree=max_degree_backward)
            if return_max_frac_err:
                X_val_pred = self.backward_emulator(Y_val)
                max_frac_err = np.max(np.abs((X_val_pred - X_val) / (X_val+1e-10)))
                self.backward_max_frac_err = max_frac_err
                print(f"Backward emulator maximum fractional error: {max_frac_err}")

    def generate_forward_emulator(self, 
                                  X_train_scaled, 
                                  Y_train_scaled,
                                  X_val_scaled,
                                  Y_val_scaled,
                                  RMSE_tol=1e-3, 
                                  init_deg=None, 
                                  max_degree=10):

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
            M, nu, multi_indices = compute_moments_vector_output(X_train_scaled, Y_train_scaled, d)
            coeffs = solve_emulator_coefficients(M, nu)

            Y_val_pred = evaluate_emulator(X_val_scaled, coeffs, multi_indices)

            RMSE_val = np.sqrt(mean_squared_error(Y_val_scaled, Y_val_pred))

            coeffs_list.append(coeffs)
            multi_indices_list.append(multi_indices)
            RMSE_val_list.append(RMSE_val)

            if RMSE_val < RMSE_tol:
                self.foward_degree = d
                print(f"Forward emulator generated with degree {d}, RMSE_val of {RMSE_val}.")
                break
            if d == max_degree:
                # find the degree with the lowest RMSE
                ind = np.argmin(RMSE_val_list)
                coeffs = coeffs_list[ind]
                multi_indices = multi_indices_list[ind]
                self.foward_degree = degree + ind
                warning(f"Maximum degree {max_degree} reached. Returning emulator with degree {degree+ind} with RMSE_val of {RMSE_val_list[ind]}.")
        
        self.forward_coeffs = coeffs
        self.forward_multi_indices = multi_indices
        
        pass

    def forward_emulator(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler_X.transform(X)
        Y_pred_scaled = evaluate_emulator(X_scaled, self.forward_coeffs, self.forward_multi_indices)
        Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)
        return Y_pred

    def generate_backward_emulator(self, 
                                   X_train_scaled, 
                                   Y_train_scaled,
                                   X_val_scaled,
                                   Y_val_scaled,
                                   RMSE_tol=1e-2, 
                                   init_deg=None, 
                                   max_degree=10):
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
            M, nu, multi_indices = compute_moments_vector_output(Y_train_scaled, X_train_scaled, d)
            coeffs = solve_emulator_coefficients(M, nu)

            X_val_pred = evaluate_emulator(Y_val_scaled, coeffs, multi_indices)

            RMSE_val = np.sqrt(mean_squared_error(X_val_scaled, X_val_pred))

            coeffs_list.append(coeffs)
            multi_indices_list.append(multi_indices)
            RMSE_val_list.append(RMSE_val)

            if RMSE_val < RMSE_tol:
                self.backward_degree = d
                print(f"Backward emulator generated with degree {d}, RMSE_val of {RMSE_val}.")
                break
            if d == max_degree:
                # find the degree with the lowest RMSE
                ind = np.argmin(RMSE_val_list)
                coeffs = coeffs_list[ind]
                multi_indices = multi_indices_list[ind]
                self.backward_degree = degree + ind
                warning(f"Maximum degree {max_degree} reached. Returning emulator with degree {degree+ind} with RMSE_val of {RMSE_val_list[ind]}.")

        self.backward_coeffs = coeffs
        self.backward_multi_indices = multi_indices

        pass

    def backward_emulator(self, Y):
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        Y_scaled = self.scaler_Y.transform(Y)
        X_pred_scaled = evaluate_emulator(Y_scaled, self.backward_coeffs, self.backward_multi_indices)
        X_pred = self.scaler_X.inverse_transform(X_pred_scaled)
        return X_pred

    def generate_forward_symb_emu(self, variable_names=None):
        exprs = symbolic_polynomial_expressions(self.forward_coeffs, 
                                                self.forward_multi_indices, 
                                                variable_names=variable_names, 
                                                input_means=self.scaler_X.mean_, 
                                                input_vars=self.scaler_X.var_,
                                                output_means=self.scaler_Y.mean_, 
                                                output_vars=self.scaler_Y.var_)
        return exprs
    
    def generate_backward_symb_emu(self, variable_names=None):
        exprs = symbolic_polynomial_expressions(self.backward_coeffs, 
                                                self.backward_multi_indices, 
                                                variable_names=variable_names, 
                                                input_means=self.scaler_Y.mean_, 
                                                input_vars=self.scaler_Y.var_,
                                                output_means=self.scaler_X.mean_,
                                                output_vars=self.scaler_X.var_)
        return exprs
