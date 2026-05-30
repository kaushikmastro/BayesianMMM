#!/usr/bin/env python3
"""
Regenerate MMM_analysis.ipynb using nbformat v4 API with strict GitHub compliance.

This script ensures the notebook is schema-compliant with nbformat 5.x and will
render correctly on GitHub with nbconvert v7.17.1.
"""

import json
import sys
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import (
    new_notebook,
    new_code_cell,
    new_markdown_cell,
)


def create_mmc_notebook():
    """Create MMM notebook using nbformat API with strict compliance."""

    # Create fresh notebook
    nb = new_notebook()

    # Add cells with explicit structure
    nb.cells.append(new_code_cell(
        source='import sys\nsys.path.insert(0, "..")\nfrom src import BayesianMMMTrainer\nimport pandas as pd\n\nprint("✓ Imports successful")'
    ))

    nb.cells.append(new_markdown_cell(
        source='# Bayesian Marketing Mix Model\n\nProduction-grade MMM using PyMC for Bayesian inference and ROI attribution.'
    ))

    nb.cells.append(new_code_cell(
        source='df = pd.read_csv("../data/dt_simulated_weekly.csv")\nprint(f"Data shape: {df.shape}")\nprint(f"Columns: {list(df.columns)}")'
    ))

    nb.cells.append(new_code_cell(
        source='config = {\n    "date_col": "DATE",\n    "spend_cols": ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"],\n    "revenue_col": "revenue",\n    "control_cols": ["competitor_sales_B", "newsletter"],\n    "fourier_k": 3,\n    "mcmc_params": {"draws": 1000, "tune": 1000, "target_accept": 0.9}\n}\n\ntrainer = BayesianMMMTrainer(config, "../data/dt_simulated_weekly.csv", "../data/dt_prophet_holidays.csv")\nprint("✓ Trainer initialized")'
    ))

    nb.cells.append(new_code_cell(
        source='trainer.load_data()\ntrainer.preprocess()\nprint("✓ Data loaded and preprocessed")\nprint(f"  Weeks: {len(trainer.data_df)}")\nprint(f"  Channels: {len(config[\'spend_cols\'])}")'
    ))

    nb.cells.append(new_markdown_cell(
        source='## Full MCMC Training\n\nFull Bayesian inference requires significant computation time. For demonstration, see `examples/quickstart.py`.'
    ))

    # Set metadata for GitHub/nbconvert compatibility
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    }

    return nb


def validate_notebook(nb, strict=True):
    """Validate notebook against nbformat schema."""
    errors = []
    warnings = []

    # Check basic structure
    if not hasattr(nb, 'cells'):
        errors.append("Missing 'cells' attribute")
        return errors, warnings

    if not hasattr(nb, 'metadata'):
        errors.append("Missing 'metadata' attribute")

    if not hasattr(nb, 'nbformat'):
        errors.append("Missing 'nbformat' version")

    # Validate each cell
    for i, cell in enumerate(nb.cells):
        cell_errors = []

        # Check required fields
        if not hasattr(cell, 'cell_type'):
            cell_errors.append(f"Cell {i}: Missing cell_type")
        elif cell.cell_type not in ['code', 'markdown', 'raw']:
            cell_errors.append(f"Cell {i}: Invalid cell_type '{cell.cell_type}'")

        # Check cell ID (required in v5.x)
        if not hasattr(cell, 'id') or not cell.id:
            cell_errors.append(f"Cell {i}: Missing or invalid id")
        elif not isinstance(cell.id, str):
            cell_errors.append(f"Cell {i}: id must be string, got {type(cell.id)}")

        # Check source
        if not hasattr(cell, 'source'):
            cell_errors.append(f"Cell {i}: Missing source")

        # Code cells must have outputs
        if cell.cell_type == 'code':
            if not hasattr(cell, 'outputs'):
                cell_errors.append(f"Cell {i}: Code cell missing outputs")
            if not hasattr(cell, 'execution_count'):
                cell_errors.append(f"Cell {i}: Code cell missing execution_count")

        if cell_errors:
            errors.extend(cell_errors)

    # Validate with nbformat
    try:
        nbf.validate(nb)
    except nbf.ValidationError as e:
        errors.append(f"nbformat ValidationError: {str(e)[:200]}")
    except Exception as e:
        if strict:
            errors.append(f"Schema validation: {type(e).__name__}: {str(e)[:200]}")
        else:
            warnings.append(f"Schema validation: {type(e).__name__}")

    return errors, warnings


def main():
    """Main entry point."""
    notebook_path = Path(__file__).parent.parent / "notebooks" / "MMM_analysis.ipynb"
    checkpoint_path = Path(__file__).parent.parent / "notebooks" / ".ipynb_checkpoints" / "MMM_analysis-checkpoint.ipynb"

    print("=" * 70)
    print("REGENERATING NOTEBOOK FOR GITHUB COMPATIBILITY")
    print("=" * 70)

    # Create notebook
    print("\n[1] Creating notebook using nbformat API...")
    nb = create_mmc_notebook()
    print(f"    ✓ Notebook created with {len(nb.cells)} cells")

    # Validate
    print("\n[2] Validating notebook structure...")
    errors, warnings = validate_notebook(nb, strict=True)

    if errors:
        print("    ✗ VALIDATION ERRORS:")
        for error in errors:
            print(f"      - {error}")
        return 1

    if warnings:
        print("    ⚠ Warnings:")
        for warning in warnings:
            print(f"      - {warning}")
    else:
        print("    ✓ Passes all validation checks")

    # Write to file
    print(f"\n[3] Writing notebook to {notebook_path.relative_to(Path.cwd())}...")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"    ✓ Written ({notebook_path.stat().st_size} bytes)")

    # Update checkpoint
    print(f"\n[4] Updating checkpoint...")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"    ✓ Checkpoint updated")

    # Final verification
    print("\n[5] Final verification...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        verify_nb = nbf.read(f, as_version=4)

    verify_errors, verify_warnings = validate_notebook(verify_nb, strict=True)

    if verify_errors:
        print("    ✗ VERIFICATION FAILED")
        for error in verify_errors:
            print(f"      - {error}")
        return 1

    print(f"    ✓ Verified: {len(verify_nb.cells)} cells, all valid")

    # JSON check
    print("\n[6] JSON schema check...")
    with open(notebook_path, 'r') as f:
        raw_json = json.load(f)

    required_keys = {'cells', 'metadata', 'nbformat', 'nbformat_minor'}
    if not required_keys.issubset(raw_json.keys()):
        print(f"    ✗ Missing keys: {required_keys - set(raw_json.keys())}")
        return 1

    print(f"    ✓ All required JSON keys present")

    print("\n" + "=" * 70)
    print("✅ NOTEBOOK REGENERATED SUCCESSFULLY")
    print("=" * 70)
    print("\nNotebook details:")
    print(f"  • Path: {notebook_path}")
    print(f"  • Size: {notebook_path.stat().st_size} bytes")
    print(f"  • Format: nbformat v{verify_nb.nbformat}.{verify_nb.nbformat_minor}")
    print(f"  • Cells: {len(verify_nb.cells)}")
    print(f"  • Status: Ready for GitHub nbconvert v7.17.1")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
