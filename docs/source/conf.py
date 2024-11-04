# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import ast
import pkgutil
import importlib
from pathlib import Path
from types import ModuleType

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
autodoc_mock_imports = [
    "cvxpy",
    "botorch",
    "clarabel",
    "gpytorch",
    "scipy",
    "matplotlib",
    "torch",
    "sklearn",
    "numpy",
]


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "VectOptAL"
copyright = "2024, Cahit Yildirim"
author = "Cahit Yildirim"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "notfound.extension",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
