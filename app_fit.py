import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from io import BytesIO

from fitting_differential_core import (  # assuming you split core logic from the notebook
    get_files,
    remove_extension,
    normal_fitting,
    normal_fitting_float,
    normal_fitting_float2,
    parameter2normal_distribution,
    distribution2extinction,
    distribution2mean_std
)

def perform_fitting(file_path, data_dir, material_path, db_dir, min_wave=400, max_wave=900, L=1.0, rate=1.0, material_density=2.33):
    """
    Performs fitting on a single CSV file and returns a plot and fitting results.
    """
    index = "Wavelength"
    file = os.path.basename(file_path)
    name = remove_extension(file)

    # Load normalized data
    normed_data = pd.read_csv(file_path)
    normed_data.columns = [index, name]
    measurement_df = normed_data.set_index(index)
    measurement_fitting = measurement_df[(measurement_df.index >= min_wave) & (measurement_df.index <= max_wave)]

    # Run fitting
    best_rss = 1e7
    best_mu = best_sigma = 0
    max_diameter = 300

    best_mu, best_sigma, best_rss = normal_fitting(50, max_diameter - 50, 10, 1, 200, 10, best_rss, measurement_fitting)
    best_mu, best_sigma, best_rss = normal_fitting(best_mu - 10, best_mu + 10, 1, best_sigma - 10, best_sigma + 10, 1, best_rss, measurement_fitting)
    best_mu, best_sigma, best_rss = normal_fitting(0, 1, 10, 1, 200, 10, best_rss, measurement_fitting)
    best_mu, best_sigma, best_rss = normal_fitting_float(best_mu - 1, best_mu + 1, 0.1, best_sigma - 1, best_sigma + 1, 0.1, best_rss, measurement_fitting)
    best_mu, best_sigma, best_rss = normal_fitting_float2(best_mu - 1, best_mu + 1, 1, best_sigma - 1, best_sigma + 1, 0.1, best_rss, measurement_fitting)

    best_distribution = parameter2normal_distribution(best_mu, best_sigma)
    best_extinction = distribution2extinction(best_distribution)
    mean, std = distribution2mean_std(best_distribution)

    # Read original spectrum for concentration estimation
    df_original = pd.read_csv(file_path, index_col=0)
    df_original = df_original[(df_original.index >= min_wave) & (df_original.index <= max_wave)]
    peak_wave = float(df_original.idxmax())
    absorbance = float(df_original.max())

    calculation_max_absorbance_perone = float(distribution2extinction(best_distribution, normalize=False).max())
    calculation_absorbance = calculation_max_absorbance_perone * 0.434 * L 
    concent = float(absorbance / calculation_absorbance)

    x = list(best_distribution.keys())
    crosssection = volume = diameter = 0
    for d in x:
        freq = best_distribution[d-1]
        diameter += freq * d
        volume += d ** 3 / 6 * math.pi * 1e-21 * freq
        crosssection += d ** 2 / 4 * math.pi * 1e-14 * freq

    mean_free_path = 1 / (concent * calculation_max_absorbance_perone)

    results = {
        "Mean Diameter (nm)": mean,
        "Sigma (nm)": std,
        "CV (%)": std / mean * 100,
        "Concentration (nM)": concent * 1e-9 * rate,
        "Mass Concentration (mg/mL)": concent * volume * material_density * 1e6 * rate,
        "Particle Volume (mL)": concent * volume * rate,
        "Total Cross Section (cm^2)": concent * crosssection * rate,
        "Scattering Cross Section (cm^2)": concent * crosssection * 4 * rate,
        "Mean Free Path (cm)": mean_free_path / rate,
        "Peak Wavelength (nm)": peak_wave,
        "Absorbance at Peak": absorbance * rate,
        "Fitting Range": f"{min_wave}-{max_wave}",
        "Rate": rate
    }

    # Plot experimental vs fitted
    fig, ax = plt.subplots()
    ax.plot(measurement_df.index, measurement_df[name], label="Experimental", color="firebrick")
    ax.plot(measurement_df.index, best_extinction.values(), label="Fitted", color="royalblue")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized Absorbance")
    ax.legend()
    ax.set_title(name)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    return buf, results
