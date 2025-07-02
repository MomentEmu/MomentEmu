import matplotlib.pyplot as plt
import corner
import numpy as np
from matplotlib import ticker as mticker



def plot_predictions_1r(true_params, pred_params, param_names=None, ft=16, n_major=3, savefig='figures/figure.pdf'):
    n_params = true_params.shape[1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    if param_names is None:
        param_names = [f"$\theta_{i}$" for i in range(n_params)]

    pad_frac = 0.05
    for i in range(n_params):
        vmin = min(true_params[:, i].min(), pred_params[:, i].min())
        vmax = max(true_params[:, i].max(), pred_params[:, i].max())
        rng  = vmax - vmin
        lo   = vmin - pad_frac * rng
        hi   = vmax + pad_frac * rng

        step = (hi-lo) / n_major
        loc  = mticker.MultipleLocator(step)

        ax = axes[i] if n_params > 1 else axes

        if i==5:
            digits = 0                           # show three decimals everywhere
            fmt = mticker.StrMethodFormatter(f'{{x:.{digits}f}}')
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
        else:
            digits = 2                           # show one decimal everywhere
            fmt = mticker.StrMethodFormatter(f'{{x:.{digits}f}}')
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)

        ax.plot(true_params[:, i], pred_params[:, i], 'o', alpha=0.6)
        ax.plot([true_params[:, i].min(), true_params[:, i].max()],
                [true_params[:, i].min(), true_params[:, i].max()], 'k--')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

                # identical tick lines:
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
        ax.set_aspect("equal", adjustable="box")  # or set_box_aspect(1)
        ax.set_xlabel(f"True", fontsize=ft)
        ax.set_ylabel(f"Predicted", fontsize=ft)
        ax.tick_params(axis='both', which='major', labelsize=ft-1)
        ax.tick_params(axis='both', which='major', labelsize=ft-2)
        ax.set_title(param_names[i], fontsize=ft+1)
        ax.grid(True)
    
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()




def plot_predictions(true_params, pred_params, param_names=None, ft=16, n_major=3, savefig='figures/figure.pdf'):
    n_params = true_params.shape[1]
    fig, axes = plt.subplots(2, n_params//2, figsize=(2 * n_params, 8))
    axes = axes.flatten()
    
    if param_names is None:
        param_names = [f"$\theta_{i}$" for i in range(n_params)]

    pad_frac = 0.05
    for i in range(n_params):
        vmin = min(true_params[:, i].min(), pred_params[:, i].min())
        vmax = max(true_params[:, i].max(), pred_params[:, i].max())
        rng  = vmax - vmin
        lo   = vmin - pad_frac * rng
        hi   = vmax + pad_frac * rng

        step = (hi-lo) / n_major
        loc  = mticker.MultipleLocator(step)

        ax = axes[i] if n_params > 1 else axes

        if i==5:
            digits = 0                           # show three decimals everywhere
            fmt = mticker.StrMethodFormatter(f'{{x:.{digits}f}}')
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
        else:
            digits = 2                           # show one decimal everywhere
            fmt = mticker.StrMethodFormatter(f'{{x:.{digits}f}}')
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)

        ax.plot(true_params[:, i], pred_params[:, i], 'o', alpha=0.6)
        ax.plot([true_params[:, i].min(), true_params[:, i].max()],
                [true_params[:, i].min(), true_params[:, i].max()], 'k--')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

                # identical tick lines:
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
        ax.set_aspect("equal", adjustable="box")  # or set_box_aspect(1)
        ax.set_xlabel(f"True", fontsize=ft)
        ax.set_ylabel(f"Predicted", fontsize=ft)
        ax.tick_params(axis='both', which='major', labelsize=ft-1)
        ax.tick_params(axis='both', which='major', labelsize=ft-2)
        ax.set_title(param_names[i], fontsize=ft+1)
        ax.grid(True)
    
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
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