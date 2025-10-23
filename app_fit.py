import os
from io import BytesIO
import zipfile

import streamlit as st
import pandas as pd

from fitting_differential_core import (
    get_files,
    remove_extension,
    run_fitting,
)

st.set_page_config(page_title="Si NP Spectrum Fitter", layout="wide")
st.title("Silicon Nanoparticle Spectrum Fitter")
st.markdown(
    "Upload CSV spectra (wavelength, intensity). The app fits each spectrum using a normal-size distribution (1–500 nm) against a precomputed extinction database (e.g., `Qext_Si_host1.33.csv`)."
)

with st.sidebar:
    st.header("Fitting Settings")
    database_dir = st.text_input(
        "Database directory (server path)", value="database",
        help="Directory containing the precomputed extinction CSVs such as `Qext_Si_host1.33.csv`."
    )

    host_file_options = []
    if os.path.isdir(database_dir):
        try:
            host_file_options = [f for f in get_files(database_dir, ".csv") if f.startswith("Qext_Si_host")]
        except Exception:
            host_file_options = []

    default_host = "Qext_Si_host1.33.csv" if "Qext_Si_host1.33.csv" in host_file_options else (host_file_options[0] if host_file_options else "")
    host_file = st.selectbox(
        "Host extinction file",
        options=host_file_options if host_file_options else [default_host],
        index=host_file_options.index(default_host) if default_host in host_file_options else 0,
        help="Choose which host medium extinction table to use."
    )

    min_wave = st.number_input("Min wavelength (nm)", min_value=200, max_value=1200, value=400, step=10)
    max_wave = st.number_input("Max wavelength (nm)", min_value=200, max_value=1200, value=800, step=10)
    st.caption("Fitting window is applied to both experimental and model spectrum.")

    st.header("Weight % Calculation")
    # 🔥 CHANGE 1: New inputs for density needed for Weight % calculation
    si_density = st.number_input(
        "Si Nanoparticle Density ($\mathbf{g/cm^3}$)", 
        value=2.33, 
        step=0.01, 
        format="%.2f",
        help="Density of Silicon (Si) nanoparticles."
    )
    host_density = st.number_input(
        "Host Density ($\mathbf{g/cm^3}$)", 
        value=1.00, 
        step=0.01, 
        format="%.2f",
        help="Density of the solvent/host medium (e.g., Water is 1.00 g/cm³)."
    )

uploaded = st.file_uploader(
    "Upload one or more CSV files (two columns: wavelength, intensity)",
    type=["csv"],
    accept_multiple_files=True,
)

if not os.path.isdir(database_dir):
    st.warning(f"Database directory '{database_dir}' not found. Please create it on the server and place files like 'Qext_Si_host1.33.csv' inside.")

results_rows = []
plots = []

if uploaded and os.path.isdir(database_dir) and host_file:
    run_btn = st.button("Run Fitting on Uploaded Files", type="primary")
    if run_btn:
        for uf in uploaded:
            try:
                content = uf.read().decode("utf-8")
            except UnicodeDecodeError:
                content = uf.read().decode("latin-1")

            name = remove_extension(uf.name)
            try:
                fig, result_df = run_fitting(
                    file_content=content,
                    name=name,
                    database_dir=database_dir,
                    host_file=host_file,
                    min_wave=int(min_wave),
                    max_wave=int(max_wave),
                )
            except Exception as e:
                st.error(f"Error fitting {uf.name}: {e}")
                continue

            # 🔥 CHANGE 2: Calculate Weight % from Volume Fraction (Vol. Frac.)
            # This is the standard conversion: Weight % = (Mass_NP / Total_Mass) * 100
            
            # --- WEIGHT % CALCULATION ---
            volume_fraction_col = 'Volume Fraction' # Assumed name for the column returned by run_fitting
            
            if volume_fraction_col in result_df.columns:
                V_frac = result_df[volume_fraction_col]
                
                # Weight Fraction (W_frac) calculation:
                # W_frac = (V_frac * rho_Si) / [(V_frac * rho_Si) + ((1 - V_frac) * rho_Host)]
                numerator = V_frac * si_density
                denominator = (V_frac * si_density) + ((1 - V_frac) * host_density)
                
                weight_fraction = numerator / denominator
                
                # Add the new column to the result DataFrame
                result_df['Weight %'] = (weight_fraction * 100).round(4).astype(str) + ' %'
            else:
                st.warning(f"Could not find required column '{volume_fraction_col}' in results for {uf.name}. Cannot calculate Weight %.")
            # --- END WEIGHT % CALCULATION ---
            
            # Show plot
            st.subheader(name)
            st.pyplot(fig)

            # Collect plot as PNG for download/zip
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=200)
            buf.seek(0)
            plots.append((f"{name}.png", buf.getvalue()))

            # Show results table (now includes Weight %)
            st.dataframe(result_df, use_container_width=True)
            results_rows.append(result_df)

        # Concatenate results
        if results_rows:
            all_results = pd.concat(results_rows, ignore_index=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                # Ensure the 'Weight %' column is handled correctly when saving to CSV
                csv_bytes = all_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results CSV",
                    data=csv_bytes,
                    file_name="fitting_results.csv",
                    mime="text/csv",
                )
            with col2:
                # ZIP of plots + results CSV
                zip_buf = BytesIO()
                with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    # add results csv
                    zf.writestr("fitting_results.csv", csv_bytes)
                    # add plots
                    for png_name, png_bytes in plots:
                        zf.writestr(png_name, png_bytes)
                zip_buf.seek(0)
                st.download_button(
                    "Download Plots + Results (ZIP)",
                    data=zip_buf.getvalue(),
                    file_name="fitting_outputs.zip",
                    mime="application/zip",
                )

st.markdown("---")
st.markdown("**Tip:** Ensure your database CSV (e.g., `Qext_Si_host1.33.csv`) has wavelength as the first column and **500** subsequent columns for diameters 1..500.")
