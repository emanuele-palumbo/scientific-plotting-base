"""
utils.py

General plotting utilities for SQE-style figures.

Based on the SQE graphical-output template:
- Calibri font
- Axis labels: 11 pt
- Legend: 10 pt
- Vector PDF + PNG at 600 dpi
- Standardized panel sizes
- Standardized SQE color palette

Usage
-----
from plot_utils import *

set_sqe_style()
fig, ax = make_panel("full_standard")

# plot...
style_axis(ax, xlabel=LABEL_FS_GHZ, ylabel=LABEL_S21_DB)
save_figure(fig, "my_figure", formats=("pdf", "png"))
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, ListedColormap, Normalize


# ==========================================================
# Unit helper
# ==========================================================

def cm_to_inch(*args):
    """Convert centimeters to inches for Matplotlib figure sizes."""
    return tuple(x / 2.54 for x in args)


# ==========================================================
# SQE reference palette
# ==========================================================

SQE_COLORS = {
    "indigo": "#224968",      # RGB 34, 73, 104
    "citrine": "#E1C516",     # RGB 225, 197, 22
    "black": "#000000",       # RGB 0, 0, 0
    "red": "#BB4430",         # RGB 187, 68, 48
    "verdigris": "#7EBDC2",   # RGB 126, 189, 194
}

SQE_COLOR_CYCLE = [
    SQE_COLORS["indigo"],
    SQE_COLORS["citrine"],
    SQE_COLORS["red"],
    SQE_COLORS["verdigris"],
    SQE_COLORS["black"],
]


def sqe_cmap(name="gold_white_blue"):
    """
    Return SQE-inspired colormaps.

    Available names
    ---------------
    "gold_white_blue" : citrine -> white -> indigo
    "blue_white_red"  : indigo -> white -> red
    "red_white_blue"  : red -> white -> indigo
    "sqe_discrete"    : listed SQE color palette
    "sqe_sequential"  : indigo -> verdigris -> citrine
    "sqe_gain"        : indigo -> verdigris -> citrine
    "sqe_diverging"   : citrine -> white -> indigo
    "sqe_phase"       : red -> white -> indigo
    "sqe_power"       : indigo -> verdigris -> citrine
    "sqe_lines"       : listed SQE line colors
    """

    if name in ("gold_white_blue", "sqe_diverging"):
        return LinearSegmentedColormap.from_list(
            "sqe_gold_white_blue",
            [SQE_COLORS["citrine"], "#FFFFFF", SQE_COLORS["indigo"]]
        )

    if name in ("blue_white_red",):
        return LinearSegmentedColormap.from_list(
            "sqe_blue_white_red",
            [SQE_COLORS["indigo"], "#FFFFFF", SQE_COLORS["red"]]
        )

    if name in ("red_white_blue", "sqe_phase"):
        return LinearSegmentedColormap.from_list(
            "sqe_red_white_blue",
            [SQE_COLORS["red"], "#FFFFFF", SQE_COLORS["indigo"]]
        )

    if name in ("sqe_sequential", "sqe_gain", "sqe_power"):
        return LinearSegmentedColormap.from_list(
            "sqe_sequential",
            [SQE_COLORS["indigo"], SQE_COLORS["verdigris"], SQE_COLORS["citrine"]]
        )

    if name in ("sqe_discrete", "sqe_lines"):
        return ListedColormap(SQE_COLOR_CYCLE, name="sqe_discrete")

    return plt.get_cmap(name)


def symmetric_norm(data, clim=None, center=0.0):
    """Symmetric diverging normalization centered at `center`."""
    data = np.asarray(data)

    if clim is not None:
        vmin, vmax = clim
    else:
        max_abs = np.nanmax(np.abs(data - center))
        vmin = center - max_abs
        vmax = center + max_abs

    return TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)


def linear_norm(data=None, clim=None):
    """Linear normalization, optionally with fixed clim."""
    if clim is not None:
        return Normalize(vmin=clim[0], vmax=clim[1])
    if data is None:
        return None
    data = np.asarray(data)
    return Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))


# ==========================================================
# SQE figure sizes
# ==========================================================

SQE_FIG_SIZES_CM = {
    # Full-page panels
    "full_large": (17.8, 15.0),
    "full_standard": (17.8, 10.0),
    "full_wide": (17.8, 8.9),

    # Single-column panels
    "single_large": (8.6, 10.0),
    "single_standard": (8.6, 6.5),
    "single_small": (8.6, 4.8),

    # Common two-panel figures from our workflow
    "full_two_panel": (17.8, 12.0),
    "full_two_panel_compact": (17.8, 10.0),
}


def get_fig_size_cm(name="full_standard"):
    """Return a named SQE figure size in cm."""
    if name not in SQE_FIG_SIZES_CM:
        raise ValueError(
            f"Unknown figure size '{name}'. Available: {list(SQE_FIG_SIZES_CM.keys())}"
        )
    return SQE_FIG_SIZES_CM[name]


def get_fig_size_in(name="full_standard"):
    """Return a named SQE figure size in inches."""
    return cm_to_inch(*get_fig_size_cm(name))


# ==========================================================
# Global style
# ==========================================================

def set_sqe_style(
    font_family="Calibri",
    fontsize_axes=11,
    fontsize_legend=10,
    linewidth=2.0,
    grid=True,
):
    """Set global Matplotlib style according to the SQE template."""
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": fontsize_axes,
        "axes.labelsize": fontsize_axes,
        "axes.titlesize": fontsize_axes,
        "xtick.labelsize": fontsize_axes,
        "ytick.labelsize": fontsize_axes,
        "legend.fontsize": fontsize_legend,
        "lines.linewidth": linewidth,
        "axes.prop_cycle": plt.cycler(color=SQE_COLOR_CYCLE),
        "axes.grid": grid,
        "grid.alpha": 0.45,
        "grid.linewidth": 0.8,
        "figure.autolayout": False,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })


# ==========================================================
# Figure factories
# ==========================================================

def make_panel(size="full_standard"):
    """Create a single-axis figure using a named SQE size."""
    fig, ax = plt.subplots(figsize=get_fig_size_in(size))
    
    return fig, ax


def make_custom_panel(width_cm=17.8, height_cm=10.0):
    """Create a single-axis figure with custom dimensions in cm."""
    fig, ax = plt.subplots(figsize=cm_to_inch(width_cm, height_cm))
    return fig, ax


def make_stacked_panel(
    nrows=2,
    size="full_two_panel",
    sharex=True,
    height_ratios=None,
):
    """Create vertically stacked axes using a named SQE size."""
    gridspec_kw = None
    if height_ratios is not None:
        gridspec_kw = {"height_ratios": height_ratios}

    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=get_fig_size_in(size),
        sharex=sharex,
        gridspec_kw=gridspec_kw,
    )
    return fig, axes


def make_side_by_side_panel(
    ncols=2,
    size="full_standard",
    sharey=True,
    width_ratios=None,
):
    """Create horizontally arranged axes using a named SQE size."""
    gridspec_kw = None
    if width_ratios is not None:
        gridspec_kw = {"width_ratios": width_ratios}

    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=get_fig_size_in(size),
        sharey=sharey,
        gridspec_kw=gridspec_kw,
    )
    return fig, axes


# ==========================================================
# Axis and colorbar styling
# ==========================================================

def style_axis(
    ax,
    xlabel=None,
    ylabel=None,
    fontsize_axes=11,
    n_xticks=None,
    n_yticks=None,
    grid=True,
):
    """Apply common SQE axis formatting."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize_axes)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize_axes)

    ax.tick_params(labelsize=fontsize_axes)
    if n_xticks is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=n_xticks))
    if n_yticks is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n_yticks))
    ax.grid(grid)


def style_legend(ax, fontsize_legend=10, **kwargs):
    """Apply common SQE legend formatting."""
    return ax.legend(fontsize=fontsize_legend, **kwargs)


def style_colorbar(cbar, label=None, fontsize_cbar=10, n_ticks=5):
    """Apply common SQE colorbar formatting."""
    if label is not None:
        cbar.set_label(label, fontsize=fontsize_cbar)
    cbar.ax.tick_params(labelsize=fontsize_cbar)
    cbar.locator = MaxNLocator(nbins=n_ticks)
    cbar.update_ticks()


def save_figure(fig, filename, formats=("pdf", "png"), dpi=600):
    """
    Save a figure in one or more formats.

    Examples
    --------
    save_figure(fig, "my_plot")
    save_figure(fig, "my_plot", formats=("png",))
    save_figure(fig, "my_plot.pdf", formats=None)
    """
    if filename is None:
        return

    if formats is None:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        return

    base = str(filename)
    for ext in formats:
        ext = ext.lower().replace(".", "")
        if base.lower().endswith(f".{ext}"):
            out = base
        else:
            out = f"{base}.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")


# ==========================================================
# Common labels
# ==========================================================

LABEL_FS_GHZ = r"$f_s$ / GHz"
LABEL_FP_GHZ = r"$f_p$ / GHz"
LABEL_FREQ_GHZ = r"f / GHz"
LABEL_P_DBM = r"$P_p$ / dBm"
LABEL_PUMP_POWER_DBM = r"$P_p$ / dBm"
LABEL_K = r"$k$ / rad$\cdot$cells$^{-1}$"
LABEL_DELTA_K = r"$\Delta k$ / rad$\cdot$cells$^{-1}$"
LABEL_DELTA_K_TOT = r"$\Delta k_{\mathrm{tot}}$ / rad$\cdot$cells$^{-1}$"
LABEL_S21_DB = r"$|S_{21}|$ / dB"
LABEL_GAIN_DB = "Gain / dB"
LABEL_FLUX = r"$\Phi_{\mathrm{ext}}/\Phi_0$"


# ==========================================================
# Common plot annotations
# ==========================================================

def add_zero_line(ax, color=None, linestyle="--", linewidth=1.0):
    if color is None:
        color = SQE_COLORS["black"]
    ax.axhline(0.0, color=color, linestyle=linestyle, linewidth=linewidth)


def add_gain_peak_lines(axs, frequency_GHz, gain, n_peaks=1, color=None):
    """Add vertical lines at the largest gain frequencies."""
    if color is None:
        color = SQE_COLORS["black"]

    frequency_GHz = np.asarray(frequency_GHz)
    gain = np.asarray(gain)

    if n_peaks == 1:
        peak_indices = [np.nanargmax(gain)]
    else:
        peak_indices = np.argsort(gain)[-n_peaks:]

    f_peaks = frequency_GHz[peak_indices]

    if not isinstance(axs, (list, tuple, np.ndarray)):
        axs = [axs]

    for fp in f_peaks:
        for ax in axs:
            ax.axvline(fp, color=color, linestyle="--", linewidth=1.2)

    return f_peaks


def add_power_line(ax, pump_power, color=None):
    if color is None:
        color = SQE_COLORS["black"]
    ax.axhline(pump_power, color=color, linestyle="--", linewidth=1.3)
