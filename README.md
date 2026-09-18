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
