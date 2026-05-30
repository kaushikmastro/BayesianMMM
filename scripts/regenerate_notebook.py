#!/usr/bin/env python3
"""Regenerate MMM_analysis.ipynb using nbformat v4 API with strict GitHub compliance.

GitHub renders notebooks via nbconvert. A notebook can be valid JSON but still be
invalid per nbformat's schema, which results in GitHub showing:

  "An error occurred" (nbformat v5.x / nbconvert v7.x)

This script rebuilds a minimal, schema-compliant notebook using nbformat.v4
helpers and validates it.

Usage:
  python scripts/regenerate_notebook.py

It overwrites:
  notebooks/MMM_analysis.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def create_mmm_notebook() -> nbf.NotebookNode:
    """Create MMM notebook using nbformat API with strict compliance."""
    nb = new_notebook()

    # Use nbformat's new_* helpers so each cell has the required fields,
    # including a stable, schema-compliant `id`.
    nb.cells = [
        new_code_cell(
            source=(
                "from pathlib import Path\n"
                "import sys\n"
                "\n"
                "# Make imports work regardless of where the notebook is launched from\n"
                "repo_root = Path.cwd()\n"
                "if (repo_root / 'src').is_dir():\n"
                "    sys.path.insert(0, str(repo_root))\n"
                "elif (repo_root.parent / 'src').is_dir():\n"
                "    sys.path.insert(0, str(repo_root.parent))\n"
                "else:\n"
                "    # Fallback: assume this notebook lives under <repo_root>/notebooks\n"
                "    sys.path.insert(0, str(Path.cwd().parent))\n"
                "\n"
                "from src import BayesianMMMTrainer\n"
                "import pandas as pd\n"
                "\n"
                "print('✓ Imports successful')\n"
            )
        ),
        new_markdown_cell(
            source=(
                "# Bayesian Marketing Mix Model\n\n"
                "Production-grade MMM using PyMC for Bayesian inference and ROI attribution."
            )
        ),
        new_code_cell(
            source=(
                "from pathlib import Path\n"
                "\n"
                "data_path = (Path.cwd() / '..' / 'data' / 'dt_simulated_weekly.csv').resolve()\n"
                "df = pd.read_csv(data_path)\n"
                "print(f'Data shape: {df.shape}')\n"
                "print(f'Columns: {list(df.columns)}')\n"
            )
        ),
        new_code_cell(
            source=(
                "from pathlib import Path\n"
                "\n"
                "config = {\n"
                "    'date_col': 'DATE',\n"
                "    'spend_cols': ['tv_S', 'ooh_S', 'print_S', 'facebook_S', 'search_S'],\n"
                "    'revenue_col': 'revenue',\n"
                "    'control_cols': ['competitor_sales_B', 'newsletter'],\n"
                "    'fourier_k': 3,\n"
                "    'mcmc_params': {'draws': 1000, 'tune': 1000, 'target_accept': 0.9}\n"
                "}\n"
                "\n"
                "data_path = (Path.cwd() / '..' / 'data' / 'dt_simulated_weekly.csv').resolve()\n"
                "holidays_path = (Path.cwd() / '..' / 'data' / 'dt_prophet_holidays.csv').resolve()\n"
                "\n"
                "trainer = BayesianMMMTrainer(config, str(data_path), str(holidays_path))\n"
                "print('✓ Trainer initialized')\n"
            )
        ),
        new_code_cell(
            source=(
                "trainer.load_data()\n"
                "trainer.preprocess()\n"
                "print('✓ Data loaded and preprocessed')\n"
                "print(f'  Weeks: {len(trainer.data_df)}')\n"
                "print(f\"  Channels: {len(config['spend_cols'])}\")\n"
            )
        ),
        new_markdown_cell(
            source=(
                "## Full MCMC Training\n\n"
                "Full Bayesian inference requires significant computation time. "
                "For demonstration, see `examples/quickstart.py`."
            )
        ),
    ]

    # Metadata for GitHub/nbconvert compatibility
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0",
        },
    }

    return nb


def validate_notebook(nb: nbf.NotebookNode) -> None:
    """Validate notebook against nbformat schema."""
    # This raises nbformat.ValidationError on failure.
    nbf.validate(nb)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "MMM_analysis.ipynb"

    nb = create_mmm_notebook()
    validate_notebook(nb)

    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    # Write using nbformat to ensure correct schema serialization
    nbf.write(nb, str(notebook_path), version=4)

    # Extra safety: ensure written file is valid JSON
    json.loads(notebook_path.read_text(encoding="utf-8"))

    print(f"Wrote notebook: {notebook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
