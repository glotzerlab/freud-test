# Change Log
The format is based on
[Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to
[Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## next

### Changed
* NeighborList `filter` method has been optimized.

## v2.4.1 - 2020-11-16

### Fixed
* Python 3.8 builds with Windows MSVC were broken due to an unrecognized CMake compiler option.
* Fixed broken documentation by overriding scikit-build options.
* RPATH on Linux is now set correctly to find TBB libraries not on the global search path.
* 2D box image calculations now return zero for the image z value.

## v2.4.0 - 2020-11-09

### Added
* The Box class has a method `contains` to determine particle membership in a box.
* NeighborList class exposes `num_points` and `num_query_points` attributes.
* `compute` method of `GaussianDensity` class has a `values` argument.
* Support for pre-commit hooks.
* Python 3.9 is supported.

### Changed
* NeighborList raises a `ValueError` instead of a `RuntimeError` if provided invalid constructor arguments.
* freud now builds using scikit-build (requires CMake).

### Deprecated
* `freud.order.Translational`

### Fixed
* Source distributions now include Cython source files.
* Hexatic order parameter (unweighted) normalizes by number of neighbors instead of the symmetry order k.
* Particles with an i-j normal vector of [0, 0, 0] are excluded from 2D Voronoi NeighborList computations for numerical stability reasons.
* Memory leak in `makeDefaultNlist` function where a NeighborList was being allocated and not freed.

## v2.3.0 - 2020-08-03

### Added
* Support for garnett 0.7.
* Custom NeighborLists can be created from a set of points using `from_points`. Distances will be calculated automatically.
* The Box class has methods `compute_distances` and `compute_all_distances` to calculate distances between arrays of points and query points.
* Hexatic can now compute 2D Minkowski Structure Metrics, using `weighted=True` along with a Voronoi NeighborList.
* Examples have been added to the Cluster, Density, Environment, and Order Modules.
* Module examples have been integrated with doctests to ensure they are up to date with API.
* SphereVoxelization class in the `density` module computes a grid of voxels occupied by spheres.
* `freud.diffraction.DiffractionPattern` class (unstable) can be used to compute 2D diffraction patterns.

### Changed
* Cython is now a required dependency (not optional). Cythonized `.cpp` files have been removed.
* An instance of GaussianDensity cannot compute 3D systems if it has been previously computed 2D systems.

### Fixed
* Histogram bin locations are computed in a more numerically stable way.
* Improved error handling of Cubatic input parameters.
* PMFTs are now properly normalized such that the pair correlation function tends to unity for an ideal gas.
* PMFTXYT uses the correct orientations when `points` and `query_points` differ.
* GaussianDensity Gaussian normalization in 2D systems has been corrected.

### Removed
* Python 3.5 is no longer supported. Python 3.6+ is required.

## v2.2.0 - 2020-02-24

### Added
* NeighborQuery objects can now create NeighborLists with neighbors sorted by bond distance.
* LocalDescriptors `compute` takes an optional maximum number of neighbors to compute for each particle.

### Fixed
* Corrected calculation of neighbor distances in the Voronoi NeighborList.
* Added finite tolerance to ensure stability of 2D Voronoi NeighborList computations.

## v2.1.0 - 2019-12-19

### Added
* The Box class has methods `center_of_mass` and `center` for periodic-aware center of mass and shifting points to center on the origin.

### Changed
* The make\_random\_box system method no longer overwrites the NumPy global random number generator state.
* The face\_orientations argument of PMFTXYZ has been renamed to equiv\_orientations and must be provided as an Mx4 array, where M is the number of symmetrically equivalent particle orientations.
* Improved documentation about query modes.
* The Voronoi class uses smarter heuristics for its voro++ block sizes, resulting in significant performance gains for large systems.

### Fixed
* The from\_box method correctly passes user provided dimensions to from\_matrix it if is called.
* Correctly recognize Ovito DataCollection objects in from\_system.
* Corrected `ClusterProperties` calculation of centers of mass in specific systems.
* Set z positions to 0 for 2D GSD systems in from\_system.
* PMFTXY and PMFTXYZ index into query orientations using the query point index instead of the point index.

## v2.0.1 - 2019-11-08

### Added
* Rewrote development documentation to match the conventions and logic in version 2.0 of the code.

### Fixed
* Automatic conversion of 2D systems from various data sources.
* Mybinder deployment works with freud v2.0.
* Minor errors in freud-examples have been corrected.

## v2.0.0 - 2019-10-31

### Added
* Ability to specify "system-like" objects that contain a box and set of points for most computes.
* NeighborLists and query arguments are now accepted on equal footing by compute methods that involve neighbor finding via the `neighbors=...` argument.
* Extensive new documentation including tutorial for new users and reference sections on crucial topics.
* Standard method for preprocessing arguments of pair computations.
* New internal ManagedArray object that allows data persistence and improves indexing in C++.
* Internal threaded storage uses the standard ManagedArray object.
* C++ Histogram class to standardize n-dimensional binning and simplify writing new methods.
* Upper bound r\_max option for number of neighbors queries.
* Lower bound r\_min option for all queries.
* Steinhardt now supports l = 0, 1.
* C++ BondHistogramCompute class encapsulates logic of histogram-based methods.
* 2D PMFTs accept quaternions as well as angles for their orientations.
* ClusterProperties computes radius of gyration from the gyration tensor for each cluster.
* `freud.data` module for generating example particle systems.
* Optional normalization for RDF, useful for small systems.
* `plot()` methods for `NeighborQuery` and `Box` objects.
* Added support for reading system data directly from MDAnalysis, garnett, gsd, HOOMD-blue, and OVITO.
* Various validation tests.

### Changed
* All compute objects that perform neighbor computations now use NeighborQuery internally.
* Neighbor-based compute methods now accept NeighborQuery (or "system-like") objects as the first argument.
* All compute objects that perform neighbor computations now loop over NeighborBond objects.
* Renamed (ref\_points, points) to (points, query\_points) to clarify their usage.
* Bond vector directionality is standardized for all computes that use it (always from query\_point to point).
* Standardized naming of various common parameters across freud such as the search distance r\_max.
* Accumulation is now performed with `compute(..., reset=False)`.
* Arrays returned to Python persist even after the compute object is destroyed or resizes its arrays.
* All class attributes are stored in the C++ members and accessed via getters wrapped as Python properties.
* Code in the freud.common has been moved to freud.util.
* NeighborQuery objects require z == 0 for all points if the box is 2D.
* Renamed several Box methods, box.ParticleBuffer is now locality.PeriodicBuffer.
* Cluster now finds connected components of the neighbor graph (the cluster cutoff distance is given through query arguments).
* Refactored and renamed attributes of Cluster and ClusterProperties modules.
* CorrelationFunction of complex inputs performs the necessary conjugation of the values before computing.
* Updated GaussianDensity constructor to accept tuples as width instead of having 2 distinct signatures.
* RDF bin centers are now strictly at the center of bins.
* RDF no longer performs parallel accumulation of cumulative counts (provided no performance gains and was substantially more complex code).
* MatchEnv has been split into separate classes for the different types of computations it is capable of performing, and these classes all use v2.0-style APIs.
* The Voronoi class was rewritten to use voro++ for vastly improved performance and correctness in edge cases.
* Improved Voronoi plotting code.
* Cubatic uses standard library random functions instead of Saru (which has been removed from the repo).
* APIs for several order parameters have been standardized.
* SolidLiquid order parameter has been completely rewritten, fixing several bugs and simplifying its C++ code.
* Steinhardt uses query arguments.
* PMFTXY2D has been renamed to PMFTXY.
* Removed unused orientations from PMFTXYZ and PMFTXY.
* PMFTXY and PMFTXYZ include the phase space volume of coordinates that are implicitly integrated out (one angle in PMFTXY, and three angles in PMFTXYZ).
* Documentation uses automodule instead of autoclass.
* Citations are now included using bibtex and sphinxcontrib-bibtex.

### Fixed
* Removed all neighbor exclusion logic from all classes, depends entirely on locality module now.
* Compute classes requiring 2D systems check the dimensionality of their input boxes.
* LinkCell nearest neighbor queries properly check the largest distance found before proceeding to next shell.
* LocalDensity uses the correct number of points/query points.
* RDF no longer forces the first bin of the PCF and first two bins of the cumulative counts to be 0.
* Steinhardt uses the ThreadStorage class and properly resets memory where needed.

### Removed
* The freud.util module.
* Python 2 is no longer supported. Python 3.5+ is required.
* LinkCell no longer exposes the internals of the cell list data structure.
* Cubatic no longer returns the per-particle tensor or the constant r4 tensor.

## v1.2.2 - 2019-08-15

### Changed
* LocalWl return values are real instead of complex.

### Fixed
* Fixed missing Condon-Shortley phase affecting LocalWl and Steinhardt Wl
  computations. This missing factor of -1 caused results for third-order (Wl)
  Steinhardt order parameters to be incorrect, shown by their lack of
  rotational invariance. This problem was introduced in v0.5.0.
* Reduced various compiler warnings.
* Possible out of bounds LinkCell access.
* RDF plots now use the provided `ax` object.

## v1.2.1 - 2019-07-26

### Changed
* Optimized performance for `RotationalAutocorrelation`.
* Added new tests for cases with two different sets of points.

### Fixed
* Fixed bug resulting in the `LocalQlNear` and `LocalWlNear` class wrongly
  using a hard instead of a soft cut-off, which may have resulted in an
  incorrect number of neighbors. This would cause incorrect results especially
  for systems with an average n-th nearest-neighbor distance smaller than
  `rmax`. This problem was introduced in v0.6.4.
* Fixed duplicate neighbors found by `LinkCell` `NeighborQuery` methods
* Corrected data in `LocalQl`, `LocalWl` documentation example
* Repeated Cubatic Order Parameter computations use the correct number of
  replicates.
* Repeated calls to `LocalQl.computeNorm` properly reset the underlying data.
* Clarified documentation for `LocalBondProjection` and `MSD`

## v1.2.0 - 2019-06-27

### Added
* Added `.plot()` method and IPython/Jupyter PNG representations for many
  classes.
* `AttributeError` is raised when one tries to access an attribute that has not
  yet been computed.
* Added `freud.parallel.getNumThreads()` method.
* New examples for integration with simulation and visualization workflows.

### Changed
* Removed extra C++ includes to speed up builds.
* The C++ style is now based on clang-format.
* Refactored C++ handling of thread-local storage.
* SolLiq order parameter computations are parallelized with TBB.
* Optimized performance of Voronoi.
* Several Box properties are now given as NumPy arrays instead of tuples.
* Box methods handling multiple vectors are parallelized with TBB.
* Eigen is now used for all matrix diagonalizations.

### Fixed
* Calling setNumThreads works correctly even if a parallel compute method has
  already been called.
* Fixed segfault with chained calls to NeighborQuery API.
* Correct `exclude_ii` logic.

### Removed
* Removed outdated `computeNList` function from `LocalDescriptors`.

## v1.1.0 - 2019-05-23

### Added
* New neighbor querying API to enable reuse of query data structures (see NeighborQuery class).
* AABBQuery (AABB tree-based neighbor finding) added to public API.
* Ability to dynamically select query method based on struct of arguments.
* All compute objects have `__repr__` and `__str__` methods defined.
* NeighborLists can be accessed as arrays of particle indices via
  `__getitem__`.
* ParticleBuffer supports different buffer sizes in x, y, z.
* Box makeCoordinates, makeFraction, getImage now support 2D arrays with
  multiple points.

### Changed
* Use constant memoryviews to prevent errors with read-only inputs.
* LocalQl is now parallelized with TBB.
* Optimized performance of RotationalAutocorrelation.
* NematicOrderParameter uses SelfAdjointEigenSolver for improved stability.
* Added build flags for Cython debugging.
* LinkCell computes cell neighbors on-demand and caches the results for
  significant speedup.

### Fixed
* Corrected type of `y_max` argument to PMFTXY2D from int to float.
* Reduce logging verbosity about array conversion.
* Fixed number of threads set upon exiting the NumThreads context manager.
* Corrected quaternion array sizes and added missing defaults in the
  documentation.
* Empty ParticleBuffers return valid array shapes for concatenation.
* Wheels are built against NumPy 1.10 for improved backwards compatibility.

## v1.0.0 - 2019-02-08

### Added
* Freshly updated README and documentation homepage.
* Moved to [GitHub](https://github.com/glotzerlab/freud).
* New msd.MSD class for computing mean-squared displacements.
* New order.RotationalAutocorrelation class.
* Cython memoryviews are now used to convert between C++ and Cython.
* New and improved freud logo.
* Internal-only AABB tree (faster for many large systems, but API is unstable).

### Changed
* Improved module documentation, especially for PMFT.
* Refactored internals of LocalQl and related classes.
* Upgraded ReadTheDocs configuration.

### Fixed
* Improved CubaticOrderParameter handling of unusable seeds.
* Fixed box error in NearestNeighbors.

### Removed
* All long-deprecated methods and classes were removed.
* Bond module removed.

## v0.11.4 - 2018-11-09

### Added
* Builds are now tested on Windows via Appveyor, though officially unsupported.

### Fixed
* Multiple user-reported issues in setup.py were resolved.
* C++ errors are handled more cleanly as Python exceptions.
* Fixed bug in SolLiq box parameters.
* Documentation corrected for NeighborList.
* Various minor compiler errors on Windows were resolved.

## v0.11.3 - 2018-10-18

### Fixed
* Linux wheels are now pushed to the real PyPI server instead of the test
  server.
* macOS deployment pyenv requires patch versions to be specified.

## v0.11.2 - 2018-10-18

### Fixed
* Error in Python versions in macOS automatic deployment.

## v0.11.1 - 2018-10-18

### Added
* PyPI builds automatically deploy for Mac and Linux.

### Changed
* macOS deployment target is now 10.12 instead of 10.9 to ensure TBB
  compatibility.
* Unwrapping positions with images is now vectorized.
* Minor documentation fixes.

### Fixed
* TBB includes were not always detected correctly by setup.py.

## v0.11.0 - 2018-09-27

### Added
* Example notebooks are now shown in the documentation.
* Many unit tests were added.
* New class: `freud.environment.LocalBondProjection`.
* `freud` is now available on the Python Package Index (PyPI) as
  `freud-analysis`.

### Changed
* Documentation was revised for several modules.
* New class `freud.box.ParticleBuffer` was adapted from the previous
  `VoronoiBuffer` to include support for triclinic boxes.
* The `bond` and `pmft` modules verify system dimensionality matches the
  coordinate system used.
* Minor optimization: arrays are reduced across threads only when necessary.

### Fixed
* NumPy arrays of lengths 2, 3, 6 are now correctly ducktyped into boxes.
* Removed internal use of deprecated code.
* C++ code using `uint` has been changed to `unsigned int`, to improve compiler
  compatibility.

### Deprecated
* In `freud.locality.LinkCell`, `computeCellList()` has been replaced by
  `compute()`.

### Removed
* The `kspace` module has been removed.

## v0.10.0 - 2018-08-27

### Added
* Codecov to track test coverage.
* Properties were added to MatchEnv, AngularSeparation, Cubatic/Nematic order
  parameters, Voronoi.

### Changed
* freud uses Cython and setup.py instead of CMake for installation.
* Properties (not get functions) are the official way to access computed
  results.
* Interface module has been improved significantly.
* density.FloatCF, density.ComplexCF, order parameter documentation is
  improved.
* Many compute methods now use points, orientations from ref\_points,
  ref\_orientations if not provided.
* Reset methods have been renamed to `reset`.

### Fixed
* `kspace` module had a missing factor of pi in the volume calculation of
  `FTsphere`.

### Deprecated
* Get functions have been deprecated.
* Setter methods have been deprecated.
* Reduce methods are called internally, so the user-facing methods have been
  deprecated.

### Removed
* GaussianDensity.resetDensity() is called internally.

## v0.9.0 - 2018-07-30

### Added
* Allow specification of rmin for LocalWl (previously was only possible for
  LocalQl).
* New environment module. Contains classes split from the order module.
* Box duck-typing: methods accepting a box argument will convert box-like
  objects into freud.box.Box objects.
* All Python/Cython code is now validated with flake8 during continuous
  integration.

### Changed
* Refactoring of LocalQl and LocalWl Steinhardt order parameters.
* MatchEnv uses BiMap instead of boost::bimap.
* All boost shared\_arrays have been replaced with std::shared\_ptr.
* Replaced boost geometry with standard containers in brute force registration
  code.
* NearestNeighbors automatically uses ref\_points as the points if points are
  not provided.
* Box::unwrap and Box::wrap return the vectors after updating.
* Everything other than true order parameters moved from Order module to
  Environment module.
* Use lambda function in parallel\_for in CorrelationFunction.
* Tests no longer depend on nose. Python's unittest is used instead.
* Vastly improved documentation clarity and correctness across all modules.
* Docstrings are now in Google format. The developer guide offers guidance for
  module authors.

### Fixed
* Fixed LocalDescriptors producing NaN's in some cases.
* Fixed cython passing C++ the default argument force\_resize to
  NeighborList::resize.
* Standardize freud.common.convert\_array error message.

### Removed
* Boost is no longer needed to build or run freud.
* Removed undocumented shapesplit module.
* Removed extra argument from TransOrderParam in C++.

## v0.8.2 - 2018-06-07

### Added
* Allow specification of maximum number of neighbors to use when computing
  LocalDescriptors

### Changed
* Using the default neighbor list with LocalDescriptors requires specifying the
  precompute argument
* Updated and improved tests
* Cleaned AngularSeparation module and documentation

## v0.8.1 - 2018-05-09

### Fixed
* Memory issue in nlist resolved

## v0.8.0 - 2018-04-06

### Added
* Voronoi neighborlist now includes periodic neighbors
* Voronoi neighborlist computes weight according to the facet area in 3D
* Box module exposes `getImage(vec)`
* Voronoi module can compute and return cell volumes/areas

### Changed
* Cluster module supports box argument in compute methods.
* Refactored C++ code to reduce extraneous #includes
* Refactored PMFT code
* Refactored box module to remove unused methods
* Resolved bug in `kspace.AnalyzeSFactor3D`

### Deprecated
* Box module `getCoordinates()` in favor of duplicate `box.makeCoordinates()`

### Removed
* Removed deprecated API for ComplexWRDF and FloatWRDF

## v0.7.0 - 2018-03-02

### Added
* Added nematic order parameter
* Added optional rmin argument to density.RDF
* Added credits file
* Wrote development guide
* Added Python interface for box periodicity

### Changed
* Various bug fixes and code cleaning
* Fixed all compile-time warnings
* Ensured PEP 8 compliance everywhere
* Minimized boost dependence
* Many documentation rewrites
* Wrote development guide
* Made tests deterministic (seeded RNGs)
* Removed deprecated Box API warnings
* Standardized numpy usage

## v0.6.4 - 2018-02-05

* Added a generic neighbor list interface
* Set up CircleCI for continuous integration
* Set up documentation on ReadTheDocs
* Added bumpversion support
* Various bug fixes
* Added python-style properties for accessing data
* Fixed issues with voronoi neighbor list

## v0.6.0

* trajectory module removed
* box constructor API updated
* PMFTXYZ API updated to take in more quaternions for `face_orientations`, or
  have a sensible default value
* NearestNeighbors:
    - over-expanding box fixed
    - strict rmax mode added
    - ability to get wrapped vectors added
    - minor updates to C-API to return full lists from C
* Addition of Bonding modules
* Addition of local environment matching

## v0.5.0

* Replace boost::shared\_array with std::shared\_ptr (C++ 11)
* Moved all tbb template classes to lambda expressions
* Moved trajectory.Box to box.Box
* trajectory is deprecated
* Fixed Bond Order Diagram and allow for global, local, or orientation
  correlation
* Added python-level voronoi calculation
* Fixed issues with compiling on OS X, including against conda python installs
* Added code to compute bonds between particles in various coordinate systems

## v0.4.1
* PMFT: Fixed issue involving binning of angles correctly
* PMFT: Fixed issue in R12 which prevented compute/accumulate from being called
  with non-flattened arrays
* PMFT: Updated xyz api to allow simpler symmetric orientations to be supplied
* PMFT: Updated pmftXY2D api
* PMFT: Histograms are properly normalized, allowing for comparison between
  systems without needing to "zero" the system
* fsph: Added library to calculate spherical harmonics via cython
* Local Descriptors: Uses fsph, updates to API
* Parallel: Added default behavior to setNumThreads and added context manager


## v0.4.0

* Add compiler flags for C++11 features
* Added Saru RNG (specifically for Cubatic Order Parameter, available to all)
* Cubatic Order Parameter
* Rank 4 tensor struct
* Environment Matching/Cluster Environment
* Shape aware fourier transform for structure factor calculation
* Added deprecation warnings; use python -W once to check warnings
* Added Change Log
* Moved all documentation in Sphinx; documentation improvements
* C++ wrapping moved from Boost to Cython
* Itercell only available in python 3.x
* PMFT:
    - XY
    - XYZ with symmetry, API updated
    - R, T1, T2
    - X, Y, T2
* viz removed (is not compatible with cython)

## No change logs prior to v0.4.0
