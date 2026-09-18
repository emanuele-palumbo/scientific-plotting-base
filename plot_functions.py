from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import re
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, mark_inset,)
from matplotlib.ticker import MaxNLocator
from plot_utils import *

def read_vector_from_h5(filename, dataset_name):
    with h5py.File(filename, "r") as f:
        if dataset_name not in f:
            raise KeyError(
                f"Dataset '{dataset_name}' not found in '{filename}'. "
                f"Available datasets: {list(f.keys())}"
            )

        data = np.asarray(f[dataset_name])

    if data.ndim != 1:
        raise ValueError(
            f"Dataset '{dataset_name}' must be 1D. Found shape {data.shape}"
        )

    return data

def find_latest_h5(folder, prefix):
    folder = Path(folder)
    files = sorted(folder.glob(f"{prefix}_*.h5"))

    if not files:
        raise FileNotFoundError(
            f"No .h5 file found in '{folder}' with prefix '{prefix}_'"
        )

    return files[-1]

def plot_linear_dispersion_relation(
    data_folder=".",
    output_folder="clean_dispersion_plots",
    freq_prefix="frequency",
    k_retta_prefix="k_retta",
    k_lin_prefix="k_lin",
    freq_dataset="frequency",
    fpump=None,          # Hz
    show_harmonics=True,
    k_retta_dataset="k_retta",
    k_lin_dataset="k_lin",
    save_name="linear_dispersion_relation",
    fig_size="single_standard",
    show=True,
):
    """
    Plot the linear dispersion relation and the reference straight line.

    The function automatically searches for:
    - frequency_*.h5
    - k_retta_*.h5
    - k_lin_*.h5

    It saves:
    - figure as PDF and PNG
    - clean data as CSV
    """

    set_sqe_style()

    data_folder = Path(data_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    freq_file = find_latest_h5(data_folder, freq_prefix)
    k_retta_file = find_latest_h5(data_folder, k_retta_prefix)
    k_lin_file = find_latest_h5(data_folder, k_lin_prefix)

    frequency = read_vector_from_h5(freq_file, freq_dataset)
    k_retta = read_vector_from_h5(k_retta_file, k_retta_dataset)
    k_lin = read_vector_from_h5(k_lin_file, k_lin_dataset)

    if not (frequency.shape == k_retta.shape == k_lin.shape):
        raise ValueError(
            "frequency, k_retta and k_lin must have the same shape. "
            f"Got {frequency.shape}, {k_retta.shape}, {k_lin.shape}"
        )

    frequency_GHz = frequency / 1e9
    delta_k = k_lin - k_retta

    fig, ax = make_panel(fig_size)

    ax.plot(
        frequency_GHz,
        k_retta,
        linestyle="--",
        color=SQE_COLORS["black"],
        label=r"Linear"
    )

    ax.plot(
        frequency_GHz,
        k_lin,
        color=SQE_COLORS["indigo"],
        label=r"$k_{0}$"
    )

    if fpump is not None and show_harmonics:

        fp_GHz = fpump / 1e9

        ax.axvline(
            fp_GHz,
            color="gray",
            linewidth=1.5,
            alpha=0.8,
            label=r"$f_p$",
        )

        ax.axvline(
            fp_GHz / 2,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=r"",
        )

        ax.axvline(
            2 * fp_GHz,
            color="gray",
            linewidth=1.5,
            alpha=0.8,
            label=r"",
        )

    style_axis(
        ax,
        xlabel=LABEL_FREQ_GHZ,
        ylabel=LABEL_K,
        n_xticks=5,
        n_yticks=5,
    )

    style_legend(ax, frameon=False)

    save_figure(
        fig,
        output_folder / save_name,
        formats=("pdf", "png"),
        dpi=600,
    )

    np.savetxt(
        output_folder / f"{save_name}_data.csv",
        np.column_stack([frequency, frequency_GHz, k_retta, k_lin, delta_k]),
        delimiter=",",
        header="frequency_Hz,frequency_GHz,k_retta,k_lin,delta_k_lin",
        comments="",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "frequency": frequency,
        "frequency_GHz": frequency_GHz,
        "k_retta": k_retta,
        "k_lin": k_lin,
        "delta_k": delta_k,
        "files": {
            "frequency": freq_file,
            "k_retta": k_retta_file,
            "k_lin": k_lin_file,
        },
        "output_folder": output_folder,
    }

def plot_reflection_transmission(
    data_folder=".",
    output_folder="analysis_outputs/scattering",
    freq_prefix="frequency",
    s11_prefix="S11",
    s21_prefix="S21",
    freq_dataset="frequency",
    s11_dataset="S11",
    s21_dataset="S21",
    fpump=None,
    save_name="reflection_transmission",
    fig_size="full_standard",
    show=True,
    xlim=None,
    ylim=None,
):
    """
    Plot raw S11 and S21 data in dB in two side-by-side panels.

    It searches for:
    - frequency_*.h5
    - S11_*.h5
    - S21_*.h5
    """

    set_sqe_style()

    data_folder = Path(data_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    freq_file = find_latest_h5(data_folder, freq_prefix)
    s11_file = find_latest_h5(data_folder, s11_prefix)
    s21_file = find_latest_h5(data_folder, s21_prefix)

    frequency = read_vector_from_h5(freq_file, freq_dataset)
    S11_raw = read_vector_from_h5(s11_file, s11_dataset)
    S21_raw = read_vector_from_h5(s21_file, s21_dataset)

    S11_dB = S_to_dB(S11_raw)
    S21_dB = S_to_dB(S21_raw)

    if not (frequency.shape == S11_dB.shape == S21_dB.shape):
        raise ValueError(
            "frequency, S11 and S21 must have the same shape. "
            f"Got {frequency.shape}, {S11_dB.shape}, {S21_dB.shape}"
        )

    frequency_GHz = frequency / 1e9

    fig, axs = make_side_by_side_panel(
        ncols=2,
        size=fig_size,
        sharey=True,
    )

    ax1, ax2 = axs

    ax1.plot(
        frequency_GHz,
        S11_dB,
        color=SQE_COLORS["red"],
    )

    ax2.plot(
        frequency_GHz,
        S21_dB,
        color=SQE_COLORS["indigo"],
    )

    for ax in axs:
        style_axis(
        ax,
        xlabel=LABEL_FREQ_GHZ,
        ylabel=r"$|S_{11}|$ / dB" if ax is ax1 else r"$|S_{21}|$ / dB",
        n_xticks=5,
        n_yticks=5,
        )

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        if fpump is not None:
            fp_GHz = fpump / 1e9

            ax.axvline(
                fp_GHz,
                color=SQE_COLORS["black"],
                alpha=0.35,
                linewidth=1.4,
                label=r"$f_p$",
            )

            ax.axvline(
                fp_GHz / 2,
                color=SQE_COLORS["black"],
                alpha=0.35,
                linestyle="--",
                linewidth=1.2,
                label=r"",
            )

            ax.axvline(
                2 * fp_GHz,
                color=SQE_COLORS["black"],
                alpha=0.35,
                linewidth=1.4,
                label=r"",
            )


    fig.tight_layout()

    save_figure(
        fig,
        output_folder / save_name,
        formats=("pdf", "png"),
        dpi=600,

    )

    np.savetxt(
        output_folder / f"{save_name}_data.csv",
        np.column_stack(
            [frequency, frequency_GHz, S11_dB, S21_dB]
        ),
        delimiter=",",
        header="frequency_Hz,frequency_GHz,S11_dB,S21_dB",
        comments="",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "frequency": frequency,
        "frequency_GHz": frequency_GHz,
        "S11_dB": S11_dB,
        "S21_dB": S21_dB,
        "files": {
            "frequency": freq_file,
            "S11": s11_file,
            "S21": s21_file,
        },
        "output_folder": output_folder,
    }

def S_to_dB(S):

    """
    Convert raw complex S-parameter data to dB.

    Equivalent to Julia:
    S_to_dB(S) = 10 * log10.(abs2.(vec(Array(S))))
    """
    S = np.ravel(np.asarray(S))
    return 10 * np.log10(np.abs(S)**2)

def Ip_to_dBm(Ip, Z0=50):
    P = 2 * np.abs(Ip)**2 * Z0
    return 10 * np.log10(P / 1e-3)

def extract_Ip_from_filename(filename):
    """
    Extract Ip from filenames like:

    S12_at_0.5623413_2026-04-17_11-25-26-405.h5

    Returns Ip as float.
    """

    filename = Path(filename).name

    match = re.search(r"_at_([-+0-9.eE]+)_", filename)

    if match is None:
        raise ValueError(
            f"Could not extract Ip from filename '{filename}'. "
            "Expected pattern like S12_at_0.5623413_2026-04-17_..."
        )

    return float(match.group(1))

def add_pump_lines(ax, fpump):
    if fpump is None:
        return

    fp_GHz = fpump / 1e9

    ax.axvline(
        fp_GHz,
        color=SQE_COLORS["black"],
        alpha=0.35,
        linewidth=1.4,
        label=r"$f_p$",
    )

    ax.axvline(
        fp_GHz / 2,
        color=SQE_COLORS["black"],
        alpha=0.35,
        linestyle="--",
        linewidth=1.2,
        label=r"",
    )

    ax.axvline(
        2 * fp_GHz,
        color=SQE_COLORS["black"],
        alpha=0.35,
        linewidth=1.4,
        label=r"",
    )

def plot_gain_vs_power(

    data_folder="analysis_inputs/gain",
    output_folder="analysis_outputs/gain",
    frequency_file=None,
    s_prefix="S12",
    freq_dataset="frequency",
    s_dataset="S12",
    fpump=None,
    Z0=50,
    save_name="gain_vs_power",
    fig_size="full_standard",
    show=True,
):
    """
    Plot gain for several pump powers.

    Expected files:
    S12_at_0.5623413_2026-04-17_11-25-26-405.h5

    The number after '_at_' is read automatically as Ip.
    """

    set_sqe_style()

    data_folder = Path(data_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if frequency_file is None:
        frequency_file = find_latest_h5(data_folder, "frequency")
    else:
        frequency_file = Path(frequency_file)

    s_files = sorted(data_folder.glob(f"{s_prefix}_at_*.h5"))

    if len(s_files) == 0:
        raise FileNotFoundError(
            f"No files found with pattern {s_prefix}_at_*.h5 in {data_folder}"
        )

    frequency = read_vector_from_h5(frequency_file, freq_dataset)
    frequency_GHz = frequency / 1e9

    Ip_values = np.array([extract_Ip_from_filename(f) for f in s_files])
    Ip_values = Ip_values * 1e-6
    pump_powers_dBm = Ip_to_dBm(Ip_values, Z0=Z0)

    sort_idx = np.argsort(pump_powers_dBm)

    s_files = [s_files[i] for i in sort_idx]
    Ip_values = Ip_values[sort_idx]
    pump_powers_dBm = pump_powers_dBm[sort_idx]

    fig, ax = make_panel(fig_size)

    gain_curves = []

    for s_file, Ip, P_dBm in zip(s_files, Ip_values, pump_powers_dBm):

        S_raw = read_vector_from_h5(s_file, s_dataset)
        gain_dB = S_to_dB(S_raw)

        if gain_dB.shape != frequency.shape:
            raise ValueError(
                f"Shape mismatch for {s_file.name}: "
                f"frequency {frequency.shape}, S data {gain_dB.shape}"
            )

        ax.plot(
            frequency_GHz,
            gain_dB,
            label=rf"$P_p = {P_dBm:.1f}$ dBm",
        )

        gain_curves.append(gain_dB)

    add_pump_lines(ax, fpump)

    style_axis(
        ax,
        xlabel=LABEL_FREQ_GHZ,
        ylabel=r"$|S_{21}|$ / dB",
        n_xticks=6,
        n_yticks=6,
    )

    style_legend(ax, frameon=False, loc="best")

    fig.tight_layout()

    save_figure(
        fig,
        output_folder / save_name,
        formats=("pdf", "png"),
        dpi=600,
    )

    gain_curves = np.asarray(gain_curves)

    header = "frequency_Hz,frequency_GHz," + ",".join(
        [f"gain_dB_{P:.2f}_dBm" for P in pump_powers_dBm]
    )

    data_to_save = np.column_stack(
        [frequency, frequency_GHz, gain_curves.T]
    )

    np.savetxt(
        output_folder / f"{save_name}_data.csv",
        data_to_save,
        delimiter=",",
        header=header,
        comments="",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "frequency": frequency,
        "frequency_GHz": frequency_GHz,
        "gain_dB": gain_curves,
        "Ip_values": Ip_values,
        "pump_powers_dBm": pump_powers_dBm,
        "files": s_files,
        "output_folder": output_folder,
    }

def plot_single_gain(
    data_folder,
    filename,
    frequency_file=None,
    s_dataset=None,
    freq_dataset="frequency",
    fpump=None,
    inset_xlim=None,
    inset_ylim=None,
    inset_loc="lower right",
    output_folder=None,
    save_name=None,
    fig_size="full_standard",
    show=True,
):

    set_sqe_style()

    data_folder = Path(data_folder)

    if s_dataset is None:

        stem = Path(filename).stem

        if stem.startswith("S21"):
            s_dataset = "S21"

        elif stem.startswith("S12"):
            s_dataset = "S12"

        elif stem.startswith("S11"):
            s_dataset = "S11"

        elif stem.startswith("S22"):
            s_dataset = "S22"

        else:
            raise ValueError(
                f"Cannot determine dataset from filename {filename}"
            )

    if frequency_file is None:
        frequency_file = find_latest_h5(
            data_folder,
            "frequency",
        )

    frequency = read_vector_from_h5(
        frequency_file,
        freq_dataset,
    )

    frequency_GHz = frequency / 1e9

    s_file = data_folder / filename

    gain_dB = S_to_dB(
        read_vector_from_h5(
            s_file,
            s_dataset,
        )
    )

    Ip = extract_Ip_from_filename(
        s_file.name
    )

    pump_power_dBm = Ip_to_dBm(
        Ip * 1e-6
    )

    fig, ax = make_panel(fig_size)
