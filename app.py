import streamlit as st
import os
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import zipfile
from io import BytesIO
import shutil

# Define directories
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "converted"
PLOTS_FOLDER = "plots"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

def clear_output_folders():
    """Delete contents of OUTPUT_FOLDER and PLOTS_FOLDER."""
    for folder in [OUTPUT_FOLDER, PLOTS_FOLDER]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                st.warning(f"Could not delete {file_path}: {e}")

# 🔥 CORRECTED FUNCTION: Uses the robust sample-detection logic and numeric filtering
def process_and_convert_tsv(file_path):
    """
    Reads a Nanodrop TSV, identifies spectra using the 'Sample' tag, 
    cleans metadata, and writes the data directly to separate CSV files.
    """
    st.info("Starting processing...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        # Read all lines and strip leading/trailing whitespace
        lines = [line.strip() for line in f.readlines()]

    # Identify positions where new samples start (case-insensitive detection)
    sample_positions = [i for i, line in enumerate(lines) if line.lower().startswith("sample")]

    if not sample_positions:
        st.error("No 'Sample' markers found. The file format might be unexpected.")
        return

    # Process each sample separately
    for i in range(len(sample_positions)):
        start_idx = sample_positions[i]
        # End index is the start of the next sample, or the end of the file
        end_idx = sample_positions[i + 1] if i + 1 < len(sample_positions) else len(lines)

        # Extract the sample name from the 'Sample' line and sanitize it
        sample_name_line = lines[start_idx].strip()
        # Use the whole line as a name, and clean it up (e.g., "Sample 1" -> "Sample_1")
        sample_name_safe = re.sub(r'\W+', '_', sample_name_line) 
        
        # Define the output CSV path
        csv_path = os.path.join(OUTPUT_FOLDER, f"{sample_name_safe}.csv")

        data_rows = []
        is_data_section = False
        
        # Process lines for the current spectrum
        for line in lines[start_idx:end_idx]:
            lower_line = line.lower()
            
            # --- Robust Data Filtering Logic ---
            # 1. Look for the header row to confirm the data section starts
            if lower_line.startswith("wavelength") or "nm" in lower_line:
                 is_data_section = True
                 continue # Skip the header row itself
            
            # 2. Skip known metadata/control lines
            if any(keyword in lower_line for keyword in ["sample", "//wlcalib", "//qspecend", "date", "am", "pm"]):
                # If these appear, stop looking for data in this block
                is_data_section = False
                continue
                
            # 3. Process lines only once we're in the confirmed data section
            if is_data_section:
                # Check for two columns and numeric content in the first two columns
                parts = line.split('\t')
                
                # Filter out empty or non-data lines within the data section
                if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                    try:
                        # CRITICAL FIX: Attempt to convert the first two elements to float.
                        # This reliably identifies numeric data and filters out non-data lines.
                        float(parts[0])
                        float(parts[1])
                        data_rows.append(parts)
                    except ValueError:
                        # Ignore lines that aren't numeric data (e.g., control lines)
                        continue
            
        # Write the data to the CSV file
        if data_rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as csv_f:
                writer = csv.writer(csv_f, delimiter=",")
                writer.writerows(data_rows)
        else:
            st.warning(f"Could not find valid data for sample: {sample_name_line}. Skipping.")


def plot_csv(csv_file_path, save_plot=False):
    
    # Use 'header=None' as the conversion function now only writes data rows
    try:
        df = pd.read_csv(csv_file_path, header=None)
    except pd.errors.EmptyDataError:
        st.warning(f"File {csv_file_path} is empty or invalid.")
        return None

    if df.shape[1] < 2:
        st.warning(f"File {csv_file_path} doesn't have at least two columns to plot (Wavelength and Extinction).")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Ensure columns 0 and 1 are numeric before plotting
    try:
        # Use coerce to turn non-numeric values (if any) into NaN, which dropna handles
        x_data = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        y_data = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        
        # Drop NaN values introduced by conversion errors
        plot_df = pd.DataFrame({'Wavelength': x_data, 'Extinction': y_data}).dropna()
        
        ax.plot(plot_df['Wavelength'], plot_df['Extinction'])
        
    except Exception as e:
        st.error(f"Error plotting data from {csv_file_path}: {e}")
        plt.close(fig) 
        return None
        
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Extinction")
    # Set the standard Nanodrop range for better visualization
    ax.set_xlim(350, 850) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(os.path.basename(csv_file_path))

    if save_plot:
        plot_path = os.path.join(PLOTS_FOLDER, Path(csv_file_path).stem + ".png")
        fig.savefig(plot_path)
        plt.close(fig) # Close figure to free up memory
        return plot_path

    plt.close(fig)
    return None # Return None if not saving

# ---------------- Streamlit App ----------------
st.title("Nanodrop .TSV to .CSV Converter (Robust)")
st.markdown("Upload a Nanodrop .tsv file and download processed .csv files. This version reliably detects and filters individual samples.")

uploaded_file = st.file_uploader("Choose a .tsv file", type=["tsv"])

if uploaded_file is not None:
    # Clear previous outputs before processing the new file
    clear_output_folders()

    # Save the uploaded file to the UPLOAD_FOLDER
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # Call the robust processing function
    process_and_convert_tsv(file_path)

    st.info("Processing complete. Download your processed files:")

    # Get the list of generated CSV files
    sorted_files = sorted([f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".csv")])

    if not sorted_files:
        st.warning("No CSV files were generated. Check the file format.")
    
    # Display download buttons and plot options for each file
    for file in sorted_files:
        csv_path = os.path.join(OUTPUT_FOLDER, file)
        col1, col2 = st.columns([3, 1])
        
        with col1:
            with open(csv_path, "rb") as f:
                st.download_button(
                    label=f"Download {file}",
                    data=f,
                    file_name=file,
                    mime="text/csv",
                    key=f"dl_{file}"
                )
        
        with col2:
            # Use unique key for each button
            plot_button_key = f"plot_{file}" 
            if st.button("Plot", key=plot_button_key):
                plot_path = plot_csv(csv_path, save_plot=True)
                if plot_path:
                    st.success(f"Plot generated for {file}")
                    # Display the plot in the app
                    st.image(plot_path, caption=file, use_column_width=True)
                    
                    # Provide a separate download button for the plot
                    with open(plot_path, "rb") as pf:
                        st.download_button(
                            label="Download Plot",
                            data=pf,
                            file_name=os.path.basename(plot_path),
                            mime="image/png",
                            key=f"dl_plot_{file}"
                        )

    # Create ZIP of all CSVs and Plots
    if sorted_files and st.button("Download All CSVs and Plots as ZIP", key="dl_all_zip"):
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            # Add CSVs
            for f in sorted_files:
                zipf.write(os.path.join(OUTPUT_FOLDER, f), arcname=f)
            # Add Plots
            for f in os.listdir(PLOTS_FOLDER):
                zipf.write(os.path.join(PLOTS_FOLDER, f), arcname=f)

        zip_buffer.seek(0)
        st.download_button(
            label="Download All as ZIP",
            data=zip_buffer,
            file_name="nanodrop_output.zip",
            mime="application/zip",
            key="final_zip_download"
        )
