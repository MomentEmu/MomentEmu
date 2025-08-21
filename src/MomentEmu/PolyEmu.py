import numpy as np
from itertools import combinations_with_replacement
from collections import Counter
import sympy as sp
from logging import warning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from MomentEmu.MomentEmu import generate_moment_products, solve_emulator_coefficients, predictive_mse_aic_bic, select_best_model, filter_modes



####### Multi-index generation and operations ####

def given_order_indices(n, d):
    """Generate all multi-indices α with total degree = d.
    
    Args:
        n: number of variables
        d: total degree

    Returns:
        list of multi-indices
    """
    indices = []
    for c in combinations_with_replacement(range(n), d):
        counter = Counter(c)
        alpha = [counter[i] for i in range(n)]
        indices.append(tuple(alpha))
    return np.array(indices)

def generate_multi_indices(n, d):
    """Generate all multi-indices α with total degree ≤ d.
    
    Args:
        n: number of variables
        d: total degree

    Returns:
        list of multi-indices
    """
    indices = []
    for deg in range(d + 1):
        for c in combinations_with_replacement(range(n), deg):
            counter = Counter(c)
            alpha = [counter[i] for i in range(n)]
            indices.append(tuple(alpha))
    return np.array(indices)

def indices_selection(multi_indices, d_vec):
    """Select multi-indices where each component is ≤ corresponding component in d_vec.
    
    Args:
        multi_indices: array of multi-indices (each row is a multi-index)
        d_vec: vector of maximum degrees for each variable
        
    Returns:
        filtered array of multi-indices
    """
    # Convert to numpy array if not already
    multi_indices = np.array(multi_indices)
    d_vec = np.array(d_vec)
    
    # Check each multi-index: all components must be ≤ corresponding d_vec components
    mask = np.all(multi_indices <= d_vec, axis=1)
    
    return multi_indices[mask]

def generate_multi_indices_with_degree_vec(d_vec):
    """Generate all multi-indices α where α[i] ≤ d_vec[i] for each variable i.
    
    Args:
        d_vec: vector of maximum degrees for each variable
        
    Returns:
        array of multi-indices
    """
    from itertools import product
    
    # Convert to numpy array if not already
    d_vec = np.array(d_vec)
    n = len(d_vec)
    
    # Generate all combinations using Cartesian product
    # For each variable i, generate range(0, d_vec[i] + 1)
    ranges = [range(d_vec[i] + 1) for i in range(n)]
    
    # Use itertools.product to get all combinations
    indices = list(product(*ranges))
    
    return np.array(indices)

####### Monomial/Polynomial functions ############
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

def compute_moments_vector_output(X, Y, multi_indices):
    """
    Vector-valued version of moment method.
    X: N x n input parameter array
    Y: N x m observable array
    multi_indices: list of multi-indices
    Returns: moment matrix Mm (D x D), moment vectors ν (D x m)
    """
    n = X.shape[1]

    Phi = evaluate_monomials_lazy(X, multi_indices)  # N x D

    Mm, nu = generate_moment_products(Phi, Y)

    return Mm, nu

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

def max_order(n_params, N_samples):
    import math
    k = 0
    while math.comb(n_params+k,k) < N_samples:
        k += 1
    return k


class PolyEmu():
    def __init__(self, 
                X, 
                Y, 
                X_test=None, 
                Y_test=None, 
                test_size=0.15, 
                RMSE_upper=1e-1,
                RMSE_lower=1e-2, 
                fRMSE_tol=1e-1,
                forward=True, 
                backward=False,
                init_deg_forward=None,  
                init_deg_backward=None, 
                max_degree_forward=None,
                max_degree_backward=None,
                dim_reduction=True,
                per_mode_thres=None,
                return_max_frac_err=False):
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

            max_deg_forward = max_order(self.n_params, X_train_scaled.shape[0])

            if max_degree_forward is None or max_degree_forward > max_deg_forward:
                max_degree_forward = max_deg_forward
                print(f"Set max_degree_forward to {max_degree_forward}. Otherwise, a higher degree will require more samples.")

            self.generate_forward_emulator(X_train_scaled, 
                                        Y_train_scaled,
                                        X_val_scaled,
                                        Y_val_scaled,
                                        RMSE_upper=RMSE_upper,
                                        RMSE_lower=RMSE_lower, 
                                        fRMSE_tol=fRMSE_tol,
                                        init_deg=init_deg_forward, 
                                        max_degree=max_degree_forward,
                                        dim_reduction=dim_reduction,
                                        per_mode_thres=per_mode_thres)
            if return_max_frac_err:
                Y_val_pred = self.forward_emulator(X_val)
                max_frac_err = np.max(np.abs((Y_val_pred - Y_val) / (Y_val+1e-10)))
                self.forward_max_frac_err = max_frac_err
                print(f"Forward emulator maximum fractional error: {max_frac_err}. (If the true value is close to 0, this value could be extremely large. This is fine.)")

        if backward:
            print("Generating backward emulator...")
            max_deg_backward = max_order(self.n_outputs, X_train_scaled.shape[0])

            if max_degree_backward is None or max_degree_backward > max_deg_backward:
                max_degree_backward = max_deg_backward
                print(f"Set max_degree_backward to {max_degree_backward}. Otherwise, a higher degree will require more samples.")

            self.generate_backward_emulator(X_train_scaled, 
                                            Y_train_scaled,
                                            X_val_scaled,
                                            Y_val_scaled,
                                            RMSE_upper=RMSE_upper,
                                            RMSE_lower=RMSE_lower, 
                                            fRMSE_tol=fRMSE_tol,
                                            init_deg=init_deg_backward, 
                                            max_degree=max_degree_backward,
                                            dim_reduction=dim_reduction,
                                            per_mode_thres=per_mode_thres)
            if return_max_frac_err:
                X_val_pred = self.backward_emulator(Y_val)
                max_frac_err = np.max(np.abs((X_val_pred - X_val) / (X_val+1e-10)))
                self.backward_max_frac_err = max_frac_err
                print(f"Backward emulator maximum fractional error: {max_frac_err}. (If the true value is close to 0, this value could be extremely large. This is fine.)")

    def generate_forward_emulator(self, 
                                  X_train_scaled, 
                                  Y_train_scaled,
                                  X_val_scaled,
                                  Y_val_scaled,
                                  RMSE_upper=0.1,
                                  RMSE_lower=1e-3, 
                                  fRMSE_tol=1e-1, 
                                  init_deg=None, 
                                  max_degree=None,
                                  dim_reduction=False,
                                  per_mode_thres=None):

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

        RMSE_val_list = []
        AIC_list = []
        BIC_list = []
        coeffs_list = []
        multi_indices_list = []

        for d in range(degree, max_degree + 1):
            if d == degree:
                multi_indices = generate_multi_indices(self.n_params, d)
            else:
                aux_indices = given_order_indices(self.n_params, d)
                multi_indices = np.concatenate((multi_indices, aux_indices), axis=0)
            M, nu = compute_moments_vector_output(X_train_scaled, Y_train_scaled, multi_indices)
            coeffs = solve_emulator_coefficients(M, nu)

            Y_val_pred = evaluate_emulator(X_val_scaled, coeffs, multi_indices)

            RMSE_val, AIC, BIC = predictive_mse_aic_bic(Y_val_scaled, Y_val_pred, multi_indices.shape[0], n_train=X_train_scaled.shape[0])
            RMSE_val_list.append(RMSE_val)
            AIC_list.append(AIC)
            BIC_list.append(BIC)

            if RMSE_val < RMSE_upper: # if the RMSE is lower than the upper bound, we accept the model, and save it for later selection
                coeffs_list.append(coeffs)
                multi_indices_list.append(multi_indices)
            else: # If the RMSE exceeds the upper bound, we reject the model straight away.
                coeffs_list.append(None) 
                multi_indices_list.append(None)

            if RMSE_val < RMSE_lower:
                self.foward_degree = d
                print(f"Forward emulator generated with degree {d}, RMSE_val of {RMSE_val}.")
                break
            if d == max_degree:
                warning(f"Maximum degree {max_degree} reached. Now choose the best fit. ")
                ind = select_best_model(RMSE_val_list, aic_list=AIC_list, bic_list=BIC_list, rmse_tol=fRMSE_tol)
                assert RMSE_val_list[ind] < RMSE_upper, "Failed: The best model has RMSE higher than the upper bound."
                coeffs = coeffs_list[ind]
                multi_indices = multi_indices_list[ind]
                self.foward_degree = degree + ind
                print(f"Forward emulator generated with degree {degree+ind}, RMSE_val of {RMSE_val_list[ind]}.")
        
        if dim_reduction:
            print("Performing dimension reduction...")
            Mm, _ = compute_moments_vector_output(X_train_scaled, Y_train_scaled, multi_indices)
            if per_mode_thres is None:
                threshold = RMSE_lower * 1e-2
            else:
                threshold = min(per_mode_thres, RMSE_lower)
            mask = filter_modes(coeffs, Mm, threshold=threshold)
            multi_indices = multi_indices[mask]
            print(f"Dimension reduced  from {coeffs.shape[0]} modes to {multi_indices.shape[0]} modes.")
            Mm, nu = compute_moments_vector_output(X_train_scaled, Y_train_scaled, multi_indices)
            coeffs = solve_emulator_coefficients(Mm, nu)

            Y_val_pred = evaluate_emulator(X_val_scaled, coeffs, multi_indices)
            RMSE_val, AIC, BIC = predictive_mse_aic_bic(Y_val_scaled, Y_val_pred, multi_indices.shape[0], n_train=X_train_scaled.shape[0])
            print(f"After the dimension reduction, the RMSE: {RMSE_val}, AIC: {AIC}, BIC: {BIC}")



        self.forward_coeffs = coeffs
        self.forward_multi_indices = multi_indices
        self.forward_RMSE_list = RMSE_val_list
        self.forward_AIC_list = AIC_list
        self.forward_BIC_list = BIC_list
        
        pass

    def forward_emulator(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler_X.transform(X)
        Y_pred_scaled = evaluate_emulator(X_scaled, self.forward_coeffs, self.forward_multi_indices)
        Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)
        if X.ndim == 1:
            Y_pred = Y_pred[0]
        return Y_pred

    def generate_backward_emulator(self, 
                                   X_train_scaled, 
                                   Y_train_scaled,
                                   X_val_scaled,
                                   Y_val_scaled,
                                   RMSE_upper=0.1,
                                   RMSE_lower=1e-2, 
                                   fRMSE_tol=1e-1, 
                                   init_deg=None, 
                                   max_degree=None,
                                   dim_reduction=False,
                                   per_mode_thres=None):
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
        AIC_list = []
        BIC_list = []
        multi_indices_list = []

        for d in range(degree, max_degree + 1):
            if d == degree:
                multi_indices = generate_multi_indices(self.n_outputs, d)
            else:
                aux_indices = given_order_indices(self.n_outputs, d)
                multi_indices = np.concatenate((multi_indices, aux_indices), axis=0)
            M, nu = compute_moments_vector_output(Y_train_scaled, X_train_scaled, multi_indices)
            coeffs = solve_emulator_coefficients(M, nu)

            X_val_pred = evaluate_emulator(Y_val_scaled, coeffs, multi_indices)

            RMSE_val, AIC, BIC = predictive_mse_aic_bic(X_val_scaled, X_val_pred, multi_indices.shape[0], n_train=Y_train_scaled.shape[0])
            RMSE_val_list.append(RMSE_val)
            AIC_list.append(AIC)
            BIC_list.append(BIC)

            if RMSE_val < RMSE_upper: # if the RMSE is lower than the upper bound, we accept the model, and save it for later selection
                coeffs_list.append(coeffs)
                multi_indices_list.append(multi_indices)
            else: # If the RMSE exceeds the upper bound, we reject the model straight away.
                coeffs_list.append(None) 
                multi_indices_list.append(None)

            if RMSE_val < RMSE_lower:
                self.backward_degree = d
                print(f"Backward emulator generated with degree {d}, RMSE_val of {RMSE_val}.")
                break
            if d == max_degree:
                warning(f"Maximum degree {max_degree} reached. Now choose the best fit. ")
                ind = select_best_model(RMSE_val_list, aic_list=AIC_list, bic_list=BIC_list, rmse_tol=fRMSE_tol)
                assert RMSE_val_list[ind] < RMSE_upper, "Failed: The best model has RMSE higher than the upper bound."
                coeffs = coeffs_list[ind]
                multi_indices = multi_indices_list[ind]
                self.backward_degree = degree + ind
                print(f"Backward emulator generated with degree {degree+ind}, RMSE_val of {RMSE_val_list[ind]}.")

        if dim_reduction:
            print("Performing dimension reduction...")
            Mm, _ = compute_moments_vector_output(Y_train_scaled, X_train_scaled, multi_indices)
            if per_mode_thres is None:
                threshold = RMSE_lower * 1e-2
            else:
                threshold = min(per_mode_thres, RMSE_lower)
            mask = filter_modes(coeffs, Mm, threshold=threshold)
            multi_indices = multi_indices[mask]
            print(f"Dimension reduced  from {coeffs.shape[0]} modes to {multi_indices.shape[0]}  modes.")
            Mm, nu = compute_moments_vector_output(Y_train_scaled, X_train_scaled, multi_indices)
            coeffs = solve_emulator_coefficients(Mm, nu)
            X_val_pred = evaluate_emulator(Y_val_scaled, coeffs, multi_indices)
            RMSE_val, AIC, BIC = predictive_mse_aic_bic(X_val_scaled, X_val_pred, multi_indices.shape[0], n_train=Y_train_scaled.shape[0])
            print(f"After the dimension reduction, the RMSE: {RMSE_val}, AIC: {AIC}, BIC: {BIC}")
            
        self.backward_coeffs = coeffs
        self.backward_multi_indices = multi_indices
        self.backward_RMSE_list = RMSE_val_list
        self.backward_AIC_list = AIC_list
        self.backward_BIC_list = BIC_list

        pass

    def backward_emulator(self, Y):
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        Y_scaled = self.scaler_Y.transform(Y)
        X_pred_scaled = evaluate_emulator(Y_scaled, self.backward_coeffs, self.backward_multi_indices)
        X_pred = self.scaler_X.inverse_transform(X_pred_scaled)
        if Y.ndim == 1:
            X_pred = X_pred[0]
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
