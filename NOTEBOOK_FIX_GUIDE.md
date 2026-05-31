# Jupyter Notebook GitHub Rendering Issues - Comprehensive Guide

## Problem Statement

GitHub's notebook renderer uses nbconvert to display `.ipynb` files. The generic error **"An error occurred ... Using nbformat vX.X.X and nbconvert vX.X.X"** occurs when the notebook JSON is syntactically valid but schema-invalid for the renderer's nbformat/nbconvert versions.

### Root Causes

1. **Missing Cell IDs** (nbformat 5.x requires unique string IDs on all cells)
2. **Invalid Metadata Structure** (missing kernel or language info)
3. **Malformed Cell Structure** (missing required fields: `cell_type`, `source`, `outputs`)
4. **Encoding Issues** (non-UTF-8 characters, invisible characters)
5. **Hand-edited JSON** (manual edits introduce subtle schema violations)
6. **Version Mismatches** (notebook created with old nbformat, rendered with new nbconvert)

## Solutions (Ranked by Effectiveness)

### ✅ Solution 1: Use nbformat API (RECOMMENDED - 100% Reliable)

Instead of manually editing JSON, regenerate using nbformat library:

```python
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

# Create using API
nb = new_notebook()
nb.cells.append(new_code_cell('print("hello")'))
nb.cells.append(new_markdown_cell('# Title'))

# Write (guarantees valid structure)
with open('notebook.ipynb', 'w') as f:
    nbf.write(nb, f)
```

**Why it works:** The API enforces schema compliance at creation time.

---

### ✅ Solution 2: Regenerate + Validate Script

Create a Python script that validates and regenerates notebooks:

```bash
# Run before committing
python3 scripts/regenerate_notebook.py
```

See `scripts/regenerate_notebook.py` in this repo for a complete implementation.

---

### ✅ Solution 3: GitHub Actions CI/CD Enforcement

Add a workflow that ensures notebooks stay valid:

```yaml
name: Validate Notebooks

on: [push, pull_request]

jobs:
  notebooks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Validate and regenerate notebooks
        run: |
          pip install nbformat nbconvert
          python3 scripts/regenerate_notebook.py
      
      - name: Check for changes
        run: |
          if ! git diff --quiet; then
            echo "Notebooks need regeneration. Run: python3 scripts/regenerate_notebook.py"
            exit 1
          fi
```

---

## Debugging Checklist

If your notebook still shows "An error occurred":

- [ ] **Check JSON validity:** `python3 -c "import json; json.load(open('notebook.ipynb'))"`
- [ ] **Check nbformat:** `python3 -c "import nbformat; nbformat.read(open('notebook.ipynb'), as_version=4)"`
- [ ] **Test nbconvert:** `nbconvert --to html notebook.ipynb`
- [ ] **Check cell IDs:** All cells must have `"id": "unique-string"`
- [ ] **Check encoding:** File must be UTF-8
- [ ] **Verify metadata:** Ensure `kernelspec` and `language_info` are present
- [ ] **Regenerate:** Use nbformat API instead of manual JSON editing

---

## Quick Fix (Copy-Paste Solution)

If you need a quick one-liner to regenerate a notebook:

```bash
python3 << 'EOF'
import nbformat as nbf
import sys

notebook_path = sys.argv[1] if len(sys.argv) > 1 else 'notebook.ipynb'

# Read current notebook
with open(notebook_path, 'r') as f:
    nb = nbf.read(f, as_version=4)

# Validate and rewrite (fixes most issues)
nbf.validate(nb)
with open(notebook_path, 'w') as f:
    nbf.write(nb, f)

print(f"✓ {notebook_path} regenerated and validated")
EOF
```

---

## Resources

- [nbformat Documentation](https://nbformat.readthedocs.io/)
- [Jupyter Notebook Format Specification](https://nbformat.readthedocs.io/en/latest/format_description.html)
- [GitHub Notebook Rendering Issues](https://github.com/github/feedback/discussions?discussions_q=notebook+error)
- [nbconvert Documentation](https://nbconvert.readthedocs.io/)

---

## This Repository's Solution

This BayesianMMM repository implements the complete solution:

1. **`scripts/regenerate_notebook.py`** - Validates and regenerates all notebooks
2. **Strict validation** - Cell IDs, metadata, schema compliance
3. **CI/CD ready** - Can be integrated into GitHub Actions
4. **Community-focused** - Transferable to other projects

### Usage

```bash
# Regenerate all notebooks
python3 scripts/regenerate_notebook.py

# Or in CI
- run: pip install nbformat
- run: python3 scripts/regenerate_notebook.py
```

---

## Prevention

**Best practices to avoid this issue:**

1. ✅ Always use nbformat API for creation
2. ✅ Never manually edit `.ipynb` JSON
3. ✅ Validate on commit: `git hook` + regenerate script
4. ✅ Test locally: `nbconvert --to html notebook.ipynb`
5. ✅ Use `.gitattributes` with nbstripout to prevent output bloat
6. ✅ Add CI validation workflow

