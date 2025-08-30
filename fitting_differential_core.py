# ==========================
# file: fitting_differential_core.py
# ==========================
import os
import math
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Helper functions (faithful to your pasted core)
# --------------------------

def sum_normalize(x):
    """Normalize a list/array so the elements sum to 1 (in place for lists)."""
    total = float(np.sum(x))
    if total == 0:
        return x
    return [xi / total for xi in x]


def get_files(directory, extension):
    return [f for f in os.listdir(directory) if f.endswith(extension)]


def remove_extension(filename):
    return os.path.splitext(filename)[0]


def normal_distribution(mu, sigma):
    """Return a length-500 discrete normal (1..500) normalized to sum=1."""
    x = np.arange(1, 501)
    if sigma == 0:
        # delta function at mu (clamp to [1, 500])
        idx = int(round(mu))
        idx = max(1, min(500, idx))
        distribution = np.zeros_like(x, dtype=float)
        distribution[idx - 1] = 1.0
        return distribution
    distribution = np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))
    return distribution / distribution.sum()


def distribution2mean_std(distribution):
    x = np.arange(1, 501)
    mean = float(np.sum(distribution * x))
    std = float(np.sqrt(np.sum(distribution * (x - mean) ** 2)))
    return mean, std


def distribution2extinction(distribution, database_dir, host_file='Qext_Si_host1.33.csv', normalize=True):
    """
    Linearly mix extinction columns (diameters 1..500) from the host file
    using the provided distribution, then optionally normalize to max=1.
    The host file must have wavelength as the first column (index_col=0)
    and 500 subsequent columns corresponding to diameters 1..500.
    """
    ext_path = os.path.join(database_dir, host_file)
    ext_data = pd.read_csv(ext_path, index_col=0)

    if ext_data.shape[1] < 500:
        raise ValueError(
            f"Host file '{host_file}' must have at least 500 columns (has {ext_data.shape[1]})."
        )

    # weighted sum of columns 1..500
    ext_spectrum = np.zeros(len(ext_data), dtype=float)
    for i in range(500):
        ext_spectrum += distribution[i] * ext_data.iloc[:, i].to_numpy()

    ext_spectrum = pd.Series(ext_spectrum, index=ext_data.index)
    if normalize:
        maxv = ext_spectrum.max()
        if maxv != 0:
            ext_spectrum = ext_spectrum / maxv
    return ext_spectrum


def extinction2figure(measured_df, fitted_series, name):
    """Plot normalized experimental vs. fitted extinction."""
    fig, ax = plt.subplots(figsize=(8, 5))
    # measured_df: a single-column DataFrame indexed by wavelength
    measured_norm = measured_df.iloc[:, 0] / measured_df.iloc[:, 0].max()
    fitted_norm = fitted_series / fitted_series.max()

    ax.plot(measured_df.index, measured_norm, label="Experimental", color='firebrick')
    ax.plot(fitted_series.index, fitted_norm, label="Fitted", color='royalblue')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized Intensity")
    ax.set_title(f"Fit for {name}")
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def run_fitting(file_content, name, database_dir, host_file='Qext_Si_host1.33.csv', min_wave=400, max_wave=800):
    """
    Core fitter (grid search over mu, sigma) exactly matching your pasted logic.
    - file_content: CSV content as string with two numeric columns: wavelength, intensity
    - name: column/name label for plotting/reporting
    - database_dir: directory containing host_file (e.g., Qext_Si_host1.33.csv)
    - host_file: which precomputed extinction matrix file to use
    - min_wave, max_wave: fitting window
    Returns: (matplotlib Figure, pandas.DataFrame with results row)
    """
    # Load and window the data
    df = pd.read_csv(StringIO(file_content), header=None)
    df = df.dropna()
    df.columns = ['wavelength', name]
    df = df.set_index('wavelength')
    df = df[(df.index >= min_wave) & (df.index <= max_wave)]

    # Grid search (coarse 10-nm step, as in your code)
    best_mu = 0
    best_sigma = 0
    best_rss = float('inf')

    for mu in range(50, 251, 10):
        for sigma in range(1, 201, 10):
            dist = normal_distribution(mu, sigma)
            extinction = distribution2extinction(dist, database_dir, host_file=host_file, normalize=True)
            rss = float(((df.iloc[:, 0] / df.iloc[:, 0].max() - extinction) ** 2).sum())
            if rss < best_rss:
                best_rss = rss
                best_mu = mu
                best_sigma = sigma

    # Best distribution and final outputs
    best_dist = normal_distribution(best_mu, best_sigma)
    extinction = distribution2extinction(best_dist, database_dir, host_file=host_file, normalize=True)
    mean, std = distribution2mean_std(best_dist)

    fig = extinction2figure(df, extinction, name)
    result_data = {
        "Filename": name,
        "Mean Diameter (nm)": round(mean, 2),
        "Std Dev (nm)": round(std, 2),
        "CV (%)": round(std / mean * 100, 2) if mean else np.nan,
        "RSS": round(best_rss, 6),
        "Best mu": best_mu,
        "Best sigma": best_sigma,
        "Host file": host_file,
        "Fit range (nm)": f"{min_wave}-{max_wave}",
    }
    return fig, pd.DataFrame([result_data])