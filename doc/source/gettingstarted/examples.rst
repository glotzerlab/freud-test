.. _examples:

========
Examples
========

Examples are provided as `Jupyter <https://jupyter.org/>`_ notebooks in a separate
`freud-examples <https://github.com/glotzerlab/freud-examples>`_ repository.
These notebooks may be launched `interactively on Binder <https://mybinder.org/v2/gh/glotzerlab/freud-examples/master?filepath=index.ipynb>`_
or downloaded and run on your own system.
Visualization of data is done via `Matplotlib <https://matplotlib.org/>`_ and `Bokeh <https://bokeh.pydata.org/>`_, unless otherwise noted.


Key concepts
============

There are a few critical concepts, algorithms, and data structures that are central to all of **freud**.
The :class:`freud.box.Box` class defines the concept of a periodic simulation box, and the :mod:`freud.locality` module defines methods for finding nearest neighbors of particles.
Since both of these are used throughout **freud**, we recommend reading the :ref:`tutorial` first, before delving into the workings of specific **freud** analysis modules.

.. toctree::
    :maxdepth: 1
    :glob:

    examples/module_intros/box*
    examples/module_intros/locality*

Analysis Modules
================

These introductory examples showcase the functionality of specific modules in **freud**, showing how they can be used to perform specific types of analyses of simulations.

.. toctree::
    :maxdepth: 1
    :glob:

    examples/module_intros/cluster*
    examples/module_intros/diffraction*
    examples/module_intros/density*
    examples/module_intros/environment*
    examples/module_intros/interface*
    examples/module_intros/order*
    examples/module_intros/pmft*

Example Analyses
================

The examples below go into greater detail about specific applications of **freud** and use cases that its analysis methods enable, such as user-defined analyses, machine learning, and data visualization.

.. toctree::
    :maxdepth: 1
    :glob:

    examples/examples/NetworkX-CNA
    examples/examples/HOOMD-MC-W6/HOOMD-MC-W6
    examples/examples/GROMACS-MDTRAJ-WATER-RDF/Compute_RDF
    examples/examples/LAMMPS-LJ-MSD/LAMMPS-LJ-MSD
    examples/examples/Using Machine Learning for Structural Identification
    examples/examples/Handling Multiple Particle Types (A-B Bonds)
    examples/examples/Calculating RDF from GSD files
    examples/examples/Calculating Strain via Voxelization
    examples/examples/Visualization with fresnel
    examples/examples/Visualization with plato
    examples/examples/Visualizing 3D Voronoi and Voxelization


Benchmarks
==========

Performance is a central consideration for **freud**. Below are some benchmarks comparing **freud** to other tools offering similar analysis methods.

.. toctree::
    :maxdepth: 1
    :glob:

    examples/examples/Benchmarking*
