.. _installation:

============
Installation
============

Installing freud
================

The **freud** library can be installed via `conda <https://conda.io/projects/conda/>`_ or pip, or compiled from source.

Install via conda
-----------------

The code below will install **freud** from `conda-forge <https://anaconda.org/conda-forge/freud>`_.

.. code-block:: bash

    conda install -c conda-forge freud

Install via pip
-----------------

The code below will install **freud** from `PyPI <https://pypi.org/project/freud-analysis/>`_.

.. code-block:: bash

    pip install freud-analysis

Compile from source
-------------------

The following are **required** for installing **freud**:

- A C++14-compliant compiler
- `Python <https://www.python.org/>`__ (>=3.6)
- `NumPy <https://www.numpy.org/>`__ (>=1.14)
- `Intel Threading Building Blocks <https://www.threadingbuildingblocks.org/>`__
- `Cython <https://cython.org/>`__ (>=0.29.14)
- `scikit-build <https://scikit-build.readthedocs.io/>`__ (>=0.10.0)
- `CMake <https://cmake.org/>`__ (>=3.6.3)

.. note::

    Depending on the generator you are using, you may require a newer version of CMake.
    In particular, on Windows Visual Studio 2017 requires at least CMake 3.7.1, while Visual Studio 2019 requires CMake 3.14.
    For more information on specific generators, see the `CMake generator documentation <https://cmake.org/cmake/help/git-stage/manual/cmake-generators.7.html>`__.

The **freud** library uses scikit-build and CMake to handle the build process itself, while the other requirements are required for compiling code in **freud**.
These requirements can be met by installing the following packages from the `conda-forge channel <https://conda-forge.org/>`__:

.. code-block:: bash

    conda install -c conda-forge tbb tbb-devel numpy cython scikit-build cmake

All requirements other than TBB can also be installed via the `Python Package Index <https://pypi.org/>`__

.. code-block:: bash

    pip install numpy cython scikit-build cmake

Wheels for tbb and tbb-devel exist on PyPI, but only for certain operating systems, so your mileage may vary.
For non-conda users, we recommend using OS-specific package managers (e.g. `Homebrew <https://brew.sh/>`__ for Mac) to install TBB.
As in the snippets above, it may be necessary to install both both a TBB and a "devel" package in order to get both the headers and the shared libraries.

The code that follows builds **freud** and installs it for all users (append ``--user`` if you wish to install it to your user site directory):

.. code-block:: bash

    git clone --recurse-submodules https://github.com/glotzerlab/freud.git
    cd freud
    python setup.py install

You can also build **freud** in place so that you can run from within the folder:

.. code-block:: bash

    # Run tests from the tests directory
    python setup.py build_ext --inplace

Building **freud** in place has certain advantages, since it does not affect your Python behavior except within the **freud** directory itself (where **freud** can be imported after building).

CMake Options
+++++++++++++

The scikit-build tool allows setup.py to accept three different sets of options separated by ``--``, where each set is provided directly to scikit-build, to CMake, or to the code generator of choice, respectively.
For example, the command ``python setup.py build_ext --inplace -- -DCOVERAGE=ON -G Ninja -- -j 4`` tell scikit-build to perform an in-place build, it tells CMake to turn on the ``COVERAGE`` option and use Ninja for compilation, and it tells Ninja to compile with 4 parallel threads.
For more information on these options, see the `scikit-build docs <scikit-build.readthedocs.io/>`__.

.. note::

    The default CMake build configuration for freud is ``ReleaseWithDocs`` (not a standard build configuration like ``Release`` or ``RelWithDebInfo``).
    On installation, ``setup.py`` assumes ``--build-type=ReleaseWithDocs`` by default if no build type is specified.
    Using this build configuration is a workaround for `this issue <https://github.com/scikit-build/scikit-build/issues/518>`__ with scikit-build and Cython embedding docstrings.

In addition to standard CMake flags, the following CMake options are available for **freud**:

.. glossary::

    \--COVERAGE
      Build the Cython files with coverage support to check unit test coverage.


The **freud** CMake configuration also respects the following environment variables (in addition to standards like ``LD_LIBRARY_PATH``).

.. glossary::

    TBBROOT
      The root directory where TBB is installed.
      Useful if TBB is installed in a non-standard location or cannot be located for some other reason.
      This variable is set by the ``tbbvars.sh`` script included with TBB when building from source.

    TBB_INCLUDE_DIR
      The directory where the TBB headers (e.g. ``tbb.h``) are located.
      Useful if TBB is installed in a non-standard location or cannot be located for some other reason.

.. note::

    **freud** makes use of git submodules. To manually update git submodules, execute:

    .. code-block:: bash

        git submodule update --init --recursive

Unit Tests
==========

The unit tests for **freud** are included in the repository and are configured to be run using the Python :mod:`unittest` library:

.. code-block:: bash

    # Run tests from the tests directory
    cd tests
    python -m unittest discover .

Note that because **freud** is designed to require installation to run (i.e. it cannot be run directly out of the build directory), importing **freud** from the root of the repository will fail because it will try and import the package folder.
As a result, unit tests must be run from outside the root directory if you wish to test the installed version of **freud**.
If you want to run tests within the root directory, you can instead build **freud** in place:

.. code-block:: bash

    # Run tests from the tests directory
    python setup.py build_ext --inplace

This build will place the necessary files alongside the **freud** source files so that **freud** can be imported from the root of the repository.

Documentation
=============

The documentation for **freud** is `hosted online at ReadTheDocs <https://freud.readthedocs.io/>`_.
You may also build the documentation yourself.

Building the documentation
--------------------------

The following are **required** for building **freud** documentation:

- `Sphinx <http://www.sphinx-doc.org/>`_
- `Read the Docs Sphinx Theme <https://sphinx-rtd-theme.readthedocs.io/>`_
- `nbsphinx <https://nbsphinx.readthedocs.io/>`_
- `jupyter_sphinx <https://jupyter-sphinx.readthedocs.io/>`_
- `sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/>`_

You can install these dependencies using conda:

.. code-block:: bash

    conda install -c conda-forge sphinx sphinx_rtd_theme nbsphinx jupyter_sphinx sphinxcontrib-bibtex

or pip:

.. code-block:: bash

    pip install sphinx sphinx-rtd-theme nbsphinx jupyter-sphinx sphinxcontrib-bibtex

To build the documentation, run the following commands in the source directory:

.. code-block:: bash

    cd doc
    make html
    # Then open build/html/index.html

To build a PDF of the documentation (requires LaTeX and/or PDFLaTeX):

.. code-block:: bash

    cd doc
    make latexpdf
    # Then open build/latex/freud.pdf
