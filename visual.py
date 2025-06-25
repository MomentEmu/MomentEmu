import matplotlib.pyplot as plt
import corner
import numpy as np

def plot_predictions(true_params, pred_params, param_names=None):
    n_params = true_params.shape[1]
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    
    if param_names is None:
        param_names = [f"$\\theta_{i}$" for i in range(n_params)]

    for i in range(n_params):
        ax = axes[i] if n_params > 1 else axes
        ax.plot(true_params[:, i], pred_params[:, i], 'o', alpha=0.6)
        ax.plot([true_params[:, i].min(), true_params[:, i].max()],
                [true_params[:, i].min(), true_params[:, i].max()], 'k--')
        ax.set_xlabel(f"True {param_names[i]}")
        ax.set_ylabel(f"Predicted {param_names[i]}")
        ax.set_title(f"{param_names[i]}")
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_residuals(true_params, pred_params, param_names=None):
    residuals = pred_params - true_params
    n_params = true_params.shape[1]
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 3))

    if param_names is None:
        param_names = [f"$\\theta_{i}$" for i in range(n_params)]

    for i in range(n_params):
        ax = axes[i] if n_params > 1 else axes
        ax.hist(residuals[:, i], bins=30, alpha=0.7)
        ax.set_title(f"Residuals: {param_names[i]}")
        ax.set_xlabel("Predicted - True")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_corner_comparison(true_params, pred_params, param_names):
    import pandas as pd
    data_true = pd.DataFrame(true_params, columns=[f"{p}_true" for p in param_names])
    data_pred = pd.DataFrame(pred_params, columns=[f"{p}_pred" for p in param_names])
    data = pd.concat([data_true, data_pred], axis=1)

    corner.corner(data[[f"{p}_true" for p in param_names]], color='blue', label_kwargs={"label": "True"})
    corner.corner(data[[f"{p}_pred" for p in param_names]], color='red', label_kwargs={"label": "Predicted"})