# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

# Add the root directory (two levels up) to sys.path
sys.path.insert(0, os.path.abspath('../..'))

# Add the parent (docs) directory (one level up) to sys.path
sys.path.insert(0, os.path.abspath('..'))

# Add the tutorial directory (e.g., `tutorial/`)
sys.path.insert(0, os.path.abspath('../../tutorial'))

# Add the examples directory (e.g., `examples/`)
sys.path.insert(0, os.path.abspath('../../examples'))

project = 'DIFFICE_jax'
copyright = '2025, Yongji Wang & Ching-Yao Lai'
author = 'Yongji Wang & Ching-Yao Lai'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser", 
              "sphinx.ext.autodoc", 
              "sphinx.ext.napoleon", 
              "sphinx.ext.mathjax", 
              "sphinx.ext.viewcode", 
              "nbsphinx"]

templates_path = ['_templates']
exclude_patterns = []


# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ['.rst', '.md']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
