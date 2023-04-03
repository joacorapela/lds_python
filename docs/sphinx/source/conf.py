# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))
print("sys.path")
print(sys.path)

# -- Project information -----------------------------------------------------

project = 'Linear Dynamical Systems'
copyright = '2022, Joaquin Rapela'
author = 'Joaquin Rapela'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode']
extensions = ['sphinx.ext.autodoc','sphinx_gallery.gen_gallery',
              'sphinx.ext.viewcode']

# sphinx-gallery

sphinx_gallery_conf = {
    'examples_dirs': '../../../examples/sphinx_gallery',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    'binder': {
        # Required keys
        'org': 'joacorapela',
        'repo': 'lds_python',
        'branch': 'gh-pages', # Can be any branch, tag, or commit hash. Use a branch that hosts your docs.
        'binderhub_url': 'https://mybinder.org', # Any URL of a binderhub deployment. Must be full URL (e.g. https://mybinder.org).
        'dependencies': '../requirements.txt',
        # Optional keys
        # 'filepath_prefix': '<prefix>' # A prefix to prepend to any filepaths in Binder links.
        # 'notebooks_dir': '<notebooks-directory-name>' # Jupyter notebooks for Binder will be copied to this directory (relative to built documentation root).
        # 'use_jupyter_lab': <bool> # Whether Binder links should start Jupyter Lab instead of the Jupyter Notebook interface.
    }
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# html_theme = 'classic'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


