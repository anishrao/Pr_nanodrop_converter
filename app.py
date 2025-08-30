import streamlit as st
import os
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import zipfile
from io import BytesIO

# Define directories
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "converted"
PLOTS_FOLDER = "plots"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

def process_tsv(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunk_size = 1333
    for i in range(len(lines) // chunk_size):
        chunk_start = i * chunk_size
        chunk_end = (i + 1) * chunk_size

        index = lines[chunk_start].strip()
        index_safe = re.sub(r'\W+', '_', index)
        output_txt = os.path.join(OUTPUT_FOLDER, f"{index_safe}.txt")

        with open(output_txt, "w", encoding="utf-8") as output:
            output.writelines(lines[chunk_start:chunk_end])

        convert_to_csv(output_txt)

def convert_to_csv(txt_path):
    with open(txt_path, "r", encoding="utf-8") as tsv_f:
        tsv = csv.reader(tsv_f, delimiter="\t")
        csv_path = txt_path.replace(".txt", ".csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as csv_f:
            writer = csv.writer(csv_f, delimiter=",")
            for i, row in enumerate(tsv):
                if i < 10 or i % 2 == 1:
                    continue
                writer.writerow(row)

    os.remove(txt_path)

def plot_csv(csv_file_path, save_plot=False):
    df = pd.read_csv(csv_file_path, header=None)
    if df.shape[1] < 2:
        st.warning(f"File {csv_file_path} doesn't have at least two columns to plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(df.iloc[:, 0], df.iloc[:, 1])
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Extinction")
    ax.set_xlim(350,850)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(os.path.basename(csv_file_path))

    if save_plot:
        plot_path = os.path.join(PLOTS_FOLDER, Path(csv_file_path).stem + ".png")
        fig.savefig(plot_path)
        return plot_path

    return fig

# Streamlit App
st.title("Nanodrop .TSV to .CSV Converter")
st.markdown("Upload a Nanodrop .tsv file and download processed .csv files.")

uploaded_file = st.file_uploader("Choose a .tsv file", type=["tsv"])

if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded successfully!")

    process_tsv(file_path)

    st.info("Download your processed files:")

    sorted_files = sorted([f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".csv")])
    plot_paths = []

    for file in sorted_files:
        csv_path = os.path.join(OUTPUT_FOLDER, file)
        col1, col2 = st.columns([3, 1])
        with col1:
            with open(csv_path, "rb") as f:
                st.download_button(
                    label=f"Download {file}",
                    data=f,
                    file_name=file,
                    mime="text/csv"
                )
        with col2:
            if st.button(f"Plot {file}"):
                plot_path = plot_csv(csv_path, save_plot=True)
                if plot_path:
                    with open(plot_path, "rb") as pf:
                        st.image(pf.read(), caption=file)
                        st.download_button(
                            label=f"Download Plot {file}",
                            data=pf,
                            file_name=os.path.basename(plot_path),
                            mime="image/png"
                        )

    # Create ZIP of all CSVs and Plots
    if st.button("Download All CSVs and Plots as ZIP"):
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for f in sorted_files:
                zipf.write(os.path.join(OUTPUT_FOLDER, f), arcname=f)
            for f in os.listdir(PLOTS_FOLDER):
                zipf.write(os.path.join(PLOTS_FOLDER, f), arcname=f)

        zip_buffer.seek(0)
        st.download_button(
            label="Download All as ZIP",
            data=zip_buffer,
            file_name="nanodrop_output.zip",
            mime="application/zip"
        )
