"""Sphinx configuration for Arepo documentation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "Arepo"
author = "Alexander Kolpakov, Aidan Rocke, and Humam al'Jammas"
copyright = "2026, Alexander Kolpakov, Aidan Rocke, and Humam al'Jammas"
release = "0.3.0"

extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
]

autosectionlabel_prefix_document = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "Arepo Mathematical Engine"

nitpicky = True
suppress_warnings = [
    "ref.term",
]
