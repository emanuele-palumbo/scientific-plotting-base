# Scientific Plotting Base

Shared plotting foundation for scientific figures.

This repository is the canonical source for the plotting style and reusable plotting functions used in analysis/figure-generation projects.

## Files

- `plot_utils.py` — common SQE plotting style: palette, figure sizes, labels, colormaps, axis helpers, and figure export.
- `plot_functions.py` — reusable higher-level plotting functions built on `plot_utils.py`.

## Typical use

Keep analysis-specific scripts and data in a separate project or ZIP, and import the plotting foundation from this repository.

```python
from plot_utils import *
from plot_functions import *
```

The files in this repository are expected to evolve. Future figures should use the current repository version unless a specific historical revision is required for reproducibility.

## Output convention

The current utilities use:

- Calibri
- 11 pt axis labels
- 10 pt legends
- SQE color palette
- vector PDF plus PNG at 600 dpi
- standardized full-width and single-column panel sizes

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

## Current and legacy colormaps

Existing colormap names keep their current definitions. Original ramps are
available alongside them with explicit names:

| Current name | Current colors | Original version |
| --- | --- | --- |
| `sqe_gain`, `sqe_power` | indigo → citrine | `sqe_gain_legacy`, `sqe_power_legacy` |
| `sqe_sequential` | indigo → verdigris | `sqe_sequential_legacy` |
| `sqe_diverging` | indigo → white → citrine | `sqe_diverging_legacy` |

The original gain/power/sequential ramp is **indigo → verdigris → citrine**,
also named `sqe_indigo_verdigris_citrine`. The original diverging ramp is
**citrine → white → indigo**, also named `sqe_citrine_white_indigo`.
Other original aliases (`gold_white_blue`, `blue_white_red`,
`red_white_blue`, `sqe_phase`, `sqe_discrete`, and `sqe_lines`) remain
available with their original colors.

Append `_r` to reverse any SQE ramp, including legacy names:

```python
gain_cmap = sqe_cmap("sqe_gain_legacy")
reversed_gain_cmap = sqe_cmap("sqe_gain_legacy_r")
reflection_cmap = sqe_cmap("YlOrBr_r")  # original Matplotlib reflection ramp
```
