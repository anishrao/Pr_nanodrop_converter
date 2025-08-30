import os
import math
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

# Helper functions
def sum_normalize(x):
    sum_all = sum(x) #sum of all components
    for i in range(len(x)): 
        x[i] /= sum_all #normalized
    return x

# x = [3, 4, 6]
# sum_normalize(x)

def get_files(directory, extension):
    return [f for f in os.listdir(directory) if f.endswith(extension)]

def remove_extension(filename):
    return os.path.splitext(filename)[0]

def normal_distribution(mu, sigma):
    x = np.arange(1, 501)
    distribution = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    return distribution / distribution.sum()

def distribution2mean_std(distribution):
    x = np.arange(1, 501)
    mean = np.sum(distribution * x)
    std = np.sqrt(np.sum(distribution * (x - mean) ** 2))
    return mean, std

def distribution2extinction(distribution, database_dir, host_file='Qext_Si_host1.33.csv', normalize=True):
    ext_data = pd.read_csv(os.path.join(database_dir, host_file), index_col=0)
    x = np.arange(1, 501)
    ext_spectrum = np.zeros(len(ext_data))
    for i in range(len(x)):
        ext_spectrum += distribution[i] * ext_data.iloc[:, i]
    ext_spectrum = pd.Series(ext_spectrum, index=ext_data.index)
    if normalize:
        ext_spectrum = ext_spectrum / ext_spectrum.max()
    return ext_spectrum

def extinction2figure(measured, fitted, name):
    fig, ax = plt.subplots()
    measured_norm = measured.iloc[:, 0] / measured.iloc[:, 0].max()
    fitted_norm = fitted / fitted.max()
    ax.plot(measured.index, measured_norm, label="Experimental", color='firebrick')
    ax.plot(fitted.index, fitted_norm, label="Fitted", color='royalblue')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized Intensity")
    ax.set_title(f"Fit for {name}")
    ax.legend()
    return fig

def run_fitting(file_content, name, material_dir, database_dir, min_wave=400, max_wave=800):
    df = pd.read_csv(StringIO(file_content))
    df = df.dropna()
    df.columns = ['wavelength', name]
    df = df.set_index('wavelength')
    df = df[(df.index >= min_wave) & (df.index <= max_wave)]

    best_mu = 0
    best_sigma = 0
    best_rss = float('inf')

    for mu in range(50, 251, 10):
        for sigma in range(1, 201, 10):
            dist = normal_distribution(mu, sigma)
            extinction = distribution2extinction(dist, database_dir)
            rss = ((df[name] / df[name].max() - extinction) ** 2).sum()
            if rss < best_rss:
                best_rss = rss
                best_mu = mu
                best_sigma = sigma

    best_dist = normal_distribution(best_mu, best_sigma)
    extinction = distribution2extinction(best_dist, database_dir)
    mean, std = distribution2mean_std(best_dist)

    fig = extinction2figure(df, extinction, name)
    result_data = {
        "Filename": name,
        "Mean Diameter (nm)": round(mean, 2),
        "Std Dev (nm)": round(std, 2),
        "CV (%)": round(std / mean * 100, 2),
        "RSS": round(best_rss, 4)
    }
    return fig, pd.DataFrame([result_data])

# for int
def normal_fitting(min_mu, max_mu, interval_mu, min_sigma, max_sigma, interval_sigma, best_rss, measurement_extinction_df):
    best_m = best_mu
    best_s = best_sigma
    
    for mu in range(min_mu, max_mu + 1, interval_mu):
        for sigma in range(min_sigma, max_sigma + 1, interval_sigma):
            #distribution = parameter2normal_distribution(mu,sigma)
            distribution = parameter2normal_distribution(mu,sigma)
            loss = distribution2cost(distribution, measurement_extinction_df)
            if(loss < best_rss):
                best_rss = loss
                best_m = mu
                best_s = sigma  
    return [best_m, best_s, best_rss]

# for float
def normal_fitting_float(min_mu, max_mu, interval_mu, min_sigma, max_sigma, interval_sigma, best_rss, measurement_extinction_df):
    best_m = best_mu
    best_s = best_sigma
    
    min_mu = min_mu * 100
    min_mu = int(min_mu)
    max_mu = max_mu * 100
    max_mu = int(max_mu)
    min_sigma = min_sigma * 100
    min_sigma = int(min_sigma)
    max_sigma = max_sigma * 100
    max_sigma = int(max_sigma)
    interval_mu = interval_mu *100
    interval_mu = int(interval_mu)
    interval_sigma = interval_sigma * 100
    interval_sigma = int(interval_sigma)
    
    for mu in range(min_mu, max_mu + 1, interval_mu):
        mu_a = float(mu) / 100
        for sigma in range(min_sigma, max_sigma + 1, interval_sigma):
            sigma_a = float(sigma) / 100
            #distribution = parameter2normal_distribution(mu_a,sigma_a)
            distribution = parameter2normal_distribution(mu_a,sigma_a)
            loss = distribution2cost(distribution, measurement_extinction_df)
            if(loss < best_rss):
                best_rss = loss
                best_m = mu_a
                best_s = sigma_a 
    return [best_m, best_s, best_rss]

def normal_fitting2(min_mu, max_mu, interval_mu, min_sigma, max_sigma, interval_sigma, best_rss, measurement_extinction_df):
    best_m = best_mu
    best_s = best_sigma
    
    for mu in range(min_mu, max_mu + 1, interval_mu):
        for sigma in range(min_sigma, max_sigma + 1, interval_sigma):
            #distribution = parameter2normal_distribution(mu,sigma)
            distribution = parameter2normal_distribution(mu,sigma)
            loss = distribution2cost2(distribution, measurement_extinction_df)
            if(loss < best_rss):
                best_rss = loss
                best_m = mu
                best_s = sigma  
    return [best_m, best_s, best_rss]

# for float
def normal_fitting_float2(min_mu, max_mu, interval_mu, min_sigma, max_sigma, interval_sigma, best_rss, measurement_extinction_df):
    best_m = best_mu
    best_s = best_sigma
    
    min_mu = min_mu * 100
    min_mu = int(min_mu)
    max_mu = max_mu * 100
    max_mu = int(max_mu)
    min_sigma = min_sigma * 100
    min_sigma = int(min_sigma)
    max_sigma = max_sigma * 100
    max_sigma = int(max_sigma)
    interval_mu = interval_mu *100
    interval_mu = int(interval_mu)
    interval_sigma = interval_sigma * 100
    interval_sigma = int(interval_sigma)
    
    for mu in range(min_mu, max_mu + 1, interval_mu):
        mu_a = float(mu) / 100
        for sigma in range(min_sigma, max_sigma + 1, interval_sigma):
            sigma_a = float(sigma) / 100
            #distribution = parameter2normal_distribution(mu_a,sigma_a)
            distribution = parameter2normal_distribution(mu_a,sigma_a)
            loss = distribution2cost2(distribution, measurement_extinction_df)
            if(loss < best_rss):
                best_rss = loss
                best_m = mu_a
                best_s = sigma_a 
    return [best_m, best_s, best_rss]

def parameter2normal_distribution(mu, sigma):
    y = []
    if(sigma == 0):
        for d in x:
            if(d == mu):
                y.append(1)
            else:
                y.append(0)
        return y
    for d in x:
        y.append(1 / math.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(d - mu) ** 2 / (2 * (sigma ** 2))))
    y = sum_normalize(y)#normalize
    return y

min_diameter = 1
max_diameter = 300
x = [int(i) for i in range(min_diameter,max_diameter,1)]

q = parameter2normal_distribution(150,10)
plt.plot(x,q)



