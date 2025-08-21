import numpy as np


####### Moment vector and matrix #################

def generate_moment_products(Phi, Y):
    """Generate moment products from evaluated basis functions Phi.
    
    Args:
        Phi: evaluated basis functions (N x D), where N is the number of samples and D is the number of basis functions.
        Y: data matrix (N x m), where m is the number of output variables.
        
    Returns:
        M: moment matrix (D x D)
        nu: moment vector (D x m)
    """
    N, D = Phi.shape
    M = (Phi.T @ Phi) / N                       # D x D
    nu = (Phi.T @ Y) / N                       # D x m
    return M, nu

def solve_emulator_coefficients(M, nu):
    """
    Solve Mc = ν for each output dimension
    
    Args:
        M: moment matrix (D x D), where D is the number of basis functions.
        nu: moment vector (D x m), where m is the number of output variables.
        
    Returns: coefficients array of shape D x m
    """
    return np.linalg.solve(M, nu)  # D x m

def filter_modes(coeffs, moment_matrix, threshold=1e-3, homogeneous=True):
    """
    Filter out modes with tiny contributions.
    
    Args:
        moment_matrix: moment matrix (D x D), where D is the number of basis functions.
        coeffs: coefficients array of shape D x m, where m is the number of variables (observables) to emulate.
        threshold: threshold for filtering out modes.
        homogeneous: all the observables use the same basis if True, otherwise allow different masks of basis functions.
        
    Returns: mask array, where True means the mode is kept. Of shape D if homogeneous, otherwise of shape D x m.
    """
    # Input validation
    if coeffs.ndim != 2:
        raise ValueError("coeffs must be a 2D array of shape (D, m)")
    if moment_matrix.ndim != 2 or moment_matrix.shape[0] != moment_matrix.shape[1]:
        raise ValueError("moment_matrix must be a square 2D array")
    if coeffs.shape[0] != moment_matrix.shape[0]:
        raise ValueError("coeffs and moment_matrix dimensions are incompatible")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    
    D, m = coeffs.shape
    
    # Total squared "energy" of each emulation function:
    # energy_list[j] = coeffs[:, j]^T @ moment_matrix @ coeffs[:, j]
    energy_list = np.einsum('ij,ij->j', coeffs, np.dot(moment_matrix, coeffs))
    
    # Handle edge case where energy is zero or negative
    if np.any(energy_list <= 0):
        # For zero or negative energy, keep all modes for safety
        if homogeneous:
            return np.ones(D, dtype=bool)
        else:
            return np.ones((D, m), dtype=bool)
    
    # Relative contribution of each mode to the total energy:
    # For mode i and observable j: (coeffs[i,j]^2 * moment_matrix[i,i]) / energy_list[j]
    moment_diag = np.diag(moment_matrix)
    relative_contribution = np.outer(moment_diag, np.ones(m)) * (coeffs**2)
    relative_contribution /= energy_list[np.newaxis, :]  # Broadcasting: (D, m) / (1, m)
    
    # Handle numerical issues
    relative_contribution = np.nan_to_num(relative_contribution, nan=0.0, posinf=0.0, neginf=0.0)
    
    if homogeneous:
        # Filter out modes with tiny contributions for all observables:
        # Keep a mode if it has significant contribution to ANY observable
        mask = np.any(relative_contribution >= threshold, axis=1)
    else:
        # Filter out modes with tiny contributions for each observable:
        # Keep modes independently for each observable
        mask = relative_contribution >= threshold
    
    return mask
    

####### Metrics, cost and penalties ##############
def metrics_and_penalties(RMSE, n, k):
    """
    Calculate AIC, AICc, and BIC. 
    Interpretation: Those are metrics defined with penalty on high dimensional representations. The lower the better.
    AIC: Akaike Information Criterion
    AICc: Corrected AIC (when n is not large when compared with k)
    BIC: Bayesian Information Criterion
    
    Args:
        RMSE: root mean squared error
        n: number of samples
        k: number of parameters
        
    Returns: AIC, AICc, BIC
    """

    AIC = 2 * (n * np.log(RMSE) + k)
    AICc = AIC + (2 * k * (k + 1)) / (n - k - 1)
    BIC = 2 * n * np.log(RMSE) + k * np.log(n)

    return AIC, AICc, BIC


def predictive_mse_aic_bic(y_test, y_pred, k, n_train=None):
    """
    Compute predictive MSE, AIC and BIC on a test set, assuming Gaussian errors.
    
    Parameters
    ----------
    y_test : array-like
        True values on the test set
    y_pred : array-like
        Predicted values on the test set
    k : int
        Number of free parameters in the model
    n_train : int, optional
        Number of training samples. If provided, used for BIC penalty.
        If None, BIC uses n_test as a fallback (heuristic).
    
    Returns
    -------
    mse : float
        Predictive MSE
    aic : float
        Predictive AIC
    bic : float
        Predictive BIC
    """
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    n_test = len(y_test)
    
    # Mean squared error on test set
    mse = np.mean((y_test - y_pred)**2)
    
    # AIC formula (up to additive constants)
    aic = n_test * np.log(mse) + 2 * k
    
    # BIC formula
    if n_train is None:
        n_bic = n_test  # fallback heuristic
    else:
        n_bic = n_train
    bic = n_bic * np.log(mse) + k * np.log(n_bic)

    rmse = np.sqrt(mse)
    
    return rmse, aic, bic


def select_best_model(rmse_list, aic_list=None, bic_list=None, rmse_tol=0.05):
    """
    Select the best model based on RMSE and optionally AIC/BIC.

    Parameters
    ----------
    rmse_list : array-like
        List of RMSE values for each model
    aic_list : array-like, optional
        List of AIC values for each model
    bic_list : array-like, optional
        List of BIC values for each model
    rmse_tol : float
        Fractional tolerance above the minimum RMSE to consider models (default 0.05 = 5%)

    Returns
    -------
    best_idx : int
        Index of the selected model
    """
    rmse = np.array(rmse_list)
    n_models = len(rmse)

    # Step 1: identify models within tolerance of lowest RMSE
    rmse_min = rmse.min()
    
    # Check for invalid RMSE values
    if not np.isfinite(rmse_min):
        # If all RMSE values are invalid, select the first model
        print("Warning: RMSE values are invalid (NaN or infinite). ")
    
    candidate_mask = rmse <= rmse_min * (1 + rmse_tol)
    candidate_idxs = np.where(candidate_mask)[0]
    
    # Safety check: if no candidates found, expand the tolerance
    if len(candidate_idxs) == 0:
        print(f"Warning: No models found within {rmse_tol*100}% tolerance. Using all finite models.")

    print(f"Candidate models within {rmse_tol*100}% of min RMSE : {candidate_idxs}")
    print(f"RMSE of candidate models : {rmse[candidate_idxs]}")

    # Step 2: among candidates, pick model with lowest complexity proxy (BIC > AIC > RMSE)
    if bic_list is not None:
        bic = np.array(bic_list)
        best_idx = candidate_idxs[np.argmin(bic[candidate_idxs])]
        print(f"Selected best model index based on BIC : {best_idx}")
    elif aic_list is not None:
        aic = np.array(aic_list)
        best_idx = candidate_idxs[np.argmin(aic[candidate_idxs])]
        print(f"Selected best model index based on AIC : {best_idx}")
    else:
        # If no complexity info, pick the model with lowest RMSE
        best_idx = candidate_idxs[np.argmin(rmse[candidate_idxs])]
        print(f"Selected best model index based on RMSE : {best_idx}")

    return best_idx





