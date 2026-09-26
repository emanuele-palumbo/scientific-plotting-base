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


def _linear_cmap(name, colors):
    """Build a continuous SQE colormap from the official palette."""
    return LinearSegmentedColormap.from_list(name, colors)


def _discrete_cmap(name, colors, n=5):
    """Build an n-level discrete SQE colormap by sampling a continuous ramp."""
    cmap = _linear_cmap(f"{name}_continuous", colors)
    return ListedColormap(cmap(np.linspace(0.0, 1.0, n)), name=name)


def sqe_cmap(name="sqe_citrine_indigo"):
    """
    Return colormaps defined by the SQE reference-palette slide.

    Official continuous ramps
    -------------------------
    "sqe_citrine_indigo"         : citrine -> indigo
    "sqe_red_verdigris"          : red -> verdigris
    "sqe_indigo_white_citrine"   : indigo -> white -> citrine
    "sqe_red_white_verdigris"    : red -> white -> verdigris
    "sqe_red_white_indigo"       : red -> white -> indigo
    "sqe_citrine_white_verdigris": citrine -> white -> verdigris
    "sqe_citrine_verdigris"      : citrine -> verdigris
    "sqe_citrine_red"            : citrine -> red
    "sqe_indigo_red"             : indigo -> red
    "sqe_indigo_verdigris"       : indigo -> verdigris

    Official discrete ramps
    -----------------------
    "sqe_citrine_indigo_5"       : five levels, citrine -> indigo
    "sqe_red_verdigris_5"        : five levels, red -> verdigris

    Line/discrete palette
    ---------------------
    "sqe_discrete", "sqe_lines"  : indigo, citrine, red, verdigris, black

    Backward-compatible aliases
    ---------------------------
    "gold_white_blue" : citrine -> white -> indigo
    "blue_white_red"  : indigo -> white -> red
    "red_white_blue"  : red -> white -> indigo
    "sqe_diverging"   : indigo -> white -> citrine
    "sqe_phase"       : red -> white -> indigo
    "sqe_sequential"  : indigo -> verdigris
    "sqe_gain"        : indigo -> citrine
    "sqe_power"       : indigo -> citrine

    Legacy ramps (before the reference-palette update)
    -------------------------------------------------
    "sqe_indigo_verdigris_citrine": indigo -> verdigris -> citrine
    "sqe_gain_legacy", "sqe_power_legacy", "sqe_sequential_legacy":
        aliases for the original three-color sequential ramp
    "sqe_citrine_white_indigo", "sqe_diverging_legacy":
        citrine -> white -> indigo (original diverging direction)

    Existing names retain their current colors. Matplotlib names such as
    "YlOrBr_r" remain supported for the original reflection plots.
    Append "_r" to any SQE name, including legacy names, to reverse it.
    """
    reverse = name.endswith("_r")
    base_name = name[:-2] if reverse else name

    C = SQE_COLORS
    maps = {
        # Exact continuous ramps shown on slide 2.
        "sqe_citrine_indigo": [C["citrine"], C["indigo"]],
        "sqe_red_verdigris": [C["red"], C["verdigris"]],
        "sqe_indigo_white_citrine": [C["indigo"], "#FFFFFF", C["citrine"]],
        "sqe_red_white_verdigris": [C["red"], "#FFFFFF", C["verdigris"]],
        "sqe_red_white_indigo": [C["red"], "#FFFFFF", C["indigo"]],
        "sqe_citrine_white_verdigris": [C["citrine"], "#FFFFFF", C["verdigris"]],
        "sqe_citrine_verdigris": [C["citrine"], C["verdigris"]],
        "sqe_citrine_red": [C["citrine"], C["red"]],
        "sqe_indigo_red": [C["indigo"], C["red"]],
        "sqe_indigo_verdigris": [C["indigo"], C["verdigris"]],

        # Original ramps, explicitly named to preserve current aliases.
        "sqe_indigo_verdigris_citrine": [C["indigo"], C["verdigris"], C["citrine"]],
        "sqe_gain_legacy": [C["indigo"], C["verdigris"], C["citrine"]],
        "sqe_power_legacy": [C["indigo"], C["verdigris"], C["citrine"]],
        "sqe_sequential_legacy": [C["indigo"], C["verdigris"], C["citrine"]],
        "sqe_citrine_white_indigo": [C["citrine"], "#FFFFFF", C["indigo"]],
        "sqe_diverging_legacy": [C["citrine"], "#FFFFFF", C["indigo"]],

        # Existing aliases retain their current definitions.
        "gold_white_blue": [C["citrine"], "#FFFFFF", C["indigo"]],
        "blue_white_red": [C["indigo"], "#FFFFFF", C["red"]],
        "red_white_blue": [C["red"], "#FFFFFF", C["indigo"]],
        "sqe_diverging": [C["indigo"], "#FFFFFF", C["citrine"]],
        "sqe_phase": [C["red"], "#FFFFFF", C["indigo"]],
        "sqe_sequential": [C["indigo"], C["verdigris"]],
        "sqe_gain": [C["indigo"], C["citrine"]],
        "sqe_power": [C["indigo"], C["citrine"]],
    }

    if base_name in ("sqe_discrete", "sqe_lines"):
        cmap = ListedColormap(SQE_COLOR_CYCLE, name="sqe_discrete")
    elif base_name == "sqe_citrine_indigo_5":
        cmap = _discrete_cmap(
            "sqe_citrine_indigo_5", [C["citrine"], C["indigo"]], n=5
        )
    elif base_name == "sqe_red_verdigris_5":
        cmap = _discrete_cmap(
            "sqe_red_verdigris_5", [C["red"], C["verdigris"]], n=5
        )
    elif base_name in maps:
        cmap = _linear_cmap(base_name, maps[base_name])
    else:
        return plt.get_cmap(name)

    return cmap.reversed(name=f"{base_name}_r") if reverse else cmap


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
