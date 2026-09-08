# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Offshore OM Risk'
copyright = '2026, Elias Tche'
author = 'Elias Tche'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
]

templates_path = ['_templates']
exclude_patterns = []
bibtex_bibfiles = ["../../Documentation.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "apa"

import os
import sys

sys.path.insert(
    0,
    os.path.abspath("../../src")
)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_theme_options = {
    "light_css_variables": {
        "font-stack":                 "Fira Sans, sans-serif",
        "font-stack--monospace":      "Fira Code, monospace",
        "color-brand-primary":        "#411454",   # IFE deep purple
        "color-brand-content":        "#0068d6",   # standard blue for links
        "color-background-primary":   "#ffffff",
        "color-background-secondary": "#f8f5fc",   # IFE lys lilla 200
        "color-highlight-on-target":  "#eae1f6",   # IFE lys lilla
        "sidebar-brand-image-height": "2rem",
    },
}
html_logo = "IFE-logo-dark-rgb.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
