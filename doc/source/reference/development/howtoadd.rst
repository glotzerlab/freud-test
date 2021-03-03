=========================
Contributing to **freud**
=========================

Code Conventions
================

Python
------

Python (and Cython) code in **freud** should follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_.

During continuous integration (CI), all Python and Cython code in **freud** is tested with `flake8 <http://flake8.pycqa.org/>`_ to ensure PEP 8 compliance.
Additionally, all CMake code is tested using `cmakelang's cmake-format <https://cmake-format.readthedocs.io/en/latest/index.html>`__.
It is strongly recommended to `set up a pre-commit hook <https://pre-commit.com/>`__ to ensure code is compliant before pushing to the repository:

.. code-block:: bash

    pip install -r requirements-precommit.txt
    pre-commit install

To manually run `pre-commit <https://pre-commit.com/>`__ for all the files present in the repository, run the following command:

.. code-block:: bash

    pre-commit run --all-files --show-diff-on-failure

Documentation is written in reStructuredText and generated using `Sphinx <http://www.sphinx-doc.org/en/stable/index.html>`_.
It should be written according to the `Google Python Style Guide <https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings>`_.
A few specific notes:

- The shapes of NumPy arrays should be documented as part of the type in the following manner::

    points ((:math:`N_{points}`, 3) :class:`numpy.ndarray`):

- Optional arguments should be documented as such within the type after the actual type, and the default value should be included within the description::

    box (:class:`freud.box.Box`, optional): Simulation box (Default value = None).


C++
---

C++ code should follow the result of running :code:`clang-format-6.0` with the style specified in the file :code:`.clang-format`.
Please refer to `Clang Format 6 <http://releases.llvm.org/6.0.1/tools/clang/docs/ClangFormatStyleOptions.html>`_ for details.

When in doubt, run :code:`clang-format -style=file FILE_WITH_YOUR_CODE` in the top directory of the **freud** repository.
If installing :code:`clang-format` is not a viable option, the :code:`check-style` step of continuous integration (CI) contains the information on the correctness of the style.

Doxygen docstrings should be used for classes, functions, etc.


Code Organization
=================

The code in **freud** is a mix of Python, Cython, and C++.
From a user's perspective, methods in **freud** correspond to ``Compute`` classes, which are contained in Python modules that group methods by topic.
To keep modules well-organized, **freud** implements the following structure:

- All C++ code is stored in the ``cpp`` folder at the root of the repository, with subdirectories corresponding to each module (e.g. ``cpp/locality``).
- Python code is stored in the ``freud`` folder at the root of the repository.
- C++ code is exposed to Python using Cython code contained in pxd files with the following convention: ``freud/_MODULENAME.pxd`` (note the preceding underscore).
- The core Cython code for modules is contained in ``freud/MODULENAME.pyx`` (no underscore).
- Generated Cython C++ code (e.g. ``freud/MODULENAME.cxx``) should not be committed during development. These files are generated using Cython when building from source, and are unnecessary when installing compiled binaries.
- If a Cython module contains code that must be imported into other Cython modules (such as the :class:`freud.box.Box` class), the ``pyx`` file must be accompanied by a ``pxd`` file with the same name: ``freud/MODULENAME.pxd`` (distinguished from ``pxd`` files used to expose C++ code by the lack of a preceding underscore). For more information on how ``pxd`` files work, see the `Cython documentation <https://cython.readthedocs.io/en/latest/src/tutorial/pxd_files.html>`_.
- All tests in **freud** are based on the Python standard :mod:`unittest` library and are contained in the ``tests`` folder. Test files are named by the convention ``tests/test_MODULENAME_CLASSNAME.py``.
- Benchmarks for **freud** are contained in the ``benchmarks`` directory and are named analogously to tests: ``benchmarks/benchmark_MODULENAME_CLASSNAME.py``.

Benchmarks
----------

Benchmarking in **freud** is performed by running the ``benchmarks/benchmarker.py`` script.
This script finds all benchmarks (using the above naming convention) and attempts to run them.
Each benchmark is defined by extending the ``Benchmark`` class defined in ``benchmarks/benchmark.py``, which provides the standard benchmarking utilities used in **freud**.
Subclasses just need to define a few methods to parameterize the benchmark, construct the **freud** object being benchmarked, and then call the relevant compute method.
Rather than describing this process in detail, we consider the benchmark for the :code:`freud.density.RDF` module as an example.

.. literalinclude:: ../../../../benchmarks/benchmark_density_RDF.py
   :language: python
   :linenos:

The ``__init__`` method defines basic parameters of the run, the ``bench_setup`` method is called to build up the :class:`~freud.density.RDF` object, and the ``bench_run`` is used to time and call ``compute``.
More examples can be found in the :code:`benchmarks` directory.
The runtime of :code:`BenchmarkDensityRDF.bench_run` will be timed for :code:`number` of times on the input sizes of :code:`Ns`.
Its runtime with respect to the number of threads will also be measured.
Benchmarks are run as a part of continuous integration, with performance comparisons between the current commit and the master branch.

Steps for Adding New Code
=========================

Once you've determined to add new code to **freud**, the first step is to create a new branch off of :code:`master`.
The process of adding code differs based on whether or not you are editing an existing module in **freud**.
Adding new methods to an existing module in **freud** requires creating the new C++ files in the ``cpp`` directory, modifying the corresponding ``_MODULENAME.pxd`` file in the ``freud`` directory, and creating a wrapper class in ``freud/MODULENAME.pyx``.
If the new methods belong in a new module, you must create the corresponding ``cpp`` directory and the ``pxd`` and ``pyx`` files accordingly.

In order for code to compile, it must be added to the relevant ``CMakeLists.txt`` file.
New C++ files for existing modules must be added to the corresponding ``cpp/MODULENAME/CMakeLists.txt`` file.
For new modules, a ``cpp/NEWMODULENAME/CMakeLists.txt`` file must be created, and in addition the new module must be added to the ``cpp/CMakeLists.txt`` file in the form of both an ``add_subdirectory`` command and addition to the ``libfreud`` library in the form of an additional source in the ``add_library`` command.
Similarly, new Cython modules must be added to the appropriate list in the ``freud/CMakeLists.txt`` file depending on whether or not there is C++ code associated with the module.
Finally, you will need to import the new module in ``freud/__init__.py`` by adding :code:`from . import MODULENAME` so that your module is usable as ``freud.MODULENAME``.

Once the code is added, appropriate tests should be added to the ``tests`` folder.
Test files are named by the convention ``tests/test_MODULENAME_CLASSNAME.py``.
The final step is updating documentation, which is contained in ``rst`` files named with the convention ``doc/source/modules/MODULENAME.rst``.
If you have added a class to an existing module, all you have to do is add that same class to the ``autosummary`` section of the corresponding ``rst`` file.
If you have created a new module, you will have to create the corresponding ``rst`` file with the summary section listing classes and functions in the module followed by a more detailed description of all classes.
All classes and functions should be documented inline in the code, which allows automatic generation of the detailed section using the ``automodule`` directive (see any of the module ``rst`` files for an example).
Finally, the new file needs to be added to ``doc/source/index.rst`` in the ``API`` section.
