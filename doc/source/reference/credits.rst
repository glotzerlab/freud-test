Credits
=======

freud Developers
----------------

The following people contributed to the development of freud.

Vyas Ramasubramani - **Lead developer**

* Ensured PEP8 compliance.
* Added CircleCI continuous integration support.
* Create environment module and refactored order module.
* Rewrote most of freud docs, including order, density, and environment modules.
* Fixed nematic order parameter.
* Add properties for accessing class members.
* Various minor bug fixes.
* Refactored PMFT code.
* Refactored Steinhardt order parameter code.
* Wrote numerous examples of freud usage.
* Rewrote most of freud tests.
* Replaced CMake-based installation with setup.py using Cython.
* Add code coverage metrics.
* Added support for installing from PyPI, including ensuring that NumPy is installed.
* Converted all docstrings to Google format, fixed various incorrect docs.
* Debugged and added rotational autocorrelation code.
* Added MSD module.
* Wrote NeighborQuery, _QueryArgs, NeighborQueryResult classes.
* Wrote neighbor iterator infrastructure.
* Wrote PairCompute and SpatialHistogram parent classes.
* Wrote ManagedArray class.
* Wrote C++ histogram-related classes.
* Initial design of freud 2.0 API (NeighborQuery objects, neighbor computations, histograms).
* Standardized neighbor API in Python to use dictionaries of arguments or NeighborList objects for all pair computations.
* Standardized all attribute access into C++ with Python properties.
* Standardized variable naming of points/query\_points across all of freud.
* Standardized vector directionality in computes.
* Enabled usage of quaternions in place of angles for orientations in 2D PMFT calculations.
* Wrote new freud 2.0 compute APIs based on neighbor\_query objects and neighbors as either dictionaries or NeighborLists.
* Rewrote MatchEnv code to fit freud 2.0 API, splitting it into 3 separate calculations and rewriting internals using NeighborQuery objects.
* Wrote tutorial and reference sections of documentation.
* Unified util and common packages.
* Rewrote all docstrings in the package for freud 2.0.
* Changed Cubatic to use Mersenne Twisters for rng.
* Moved all citations into Bibtex format.
* Created data module.
* Standardized PMFT normalization.
* Enabled optional normalization of RDF.
* Changed correlation function to properly take the complex conjugate of inputs.
* Wrote developer documentation for version 2.0.
* Fixed handling of 2D systems from various data sources.
* Fixed usage of query orientations in PMFTXY, PMFTXYT and PMFTXYZ when points and query points are not identical.
* Refactored and standardized PMFT tests.
* Rewrote build system to use scikit-build.
* Added support for pre-commit hooks.

Bradley Dice - **Lead developer**

* Cleaned up various docstrings.
* Fixed bugs in HexOrderParameter.
* Cleaned up testing code.
* Added bumpversion support.
* Reduced all compile warnings.
* Added Python interface for box periodicity.
* Added Voronoi support for neighbor lists across periodic boundaries.
* Added Voronoi weights for 3D.
* Added Voronoi cell volume computation.
* Incorporated internal BiMap class for Boost removal.
* Wrote numerous examples of freud usage.
* Added some freud tests.
* Added ReadTheDocs support.
* Rewrote interface module into pure Cython.
* Added box duck-typing.
* Removed nose from unit testing.
* Use lambda function for parallelizing CorrelationFunction with TBB.
* Finalized boost removal.
* Wrote AABBQuery class.
* Consolidated cluster module functionality.
* Rewrote SolidLiquid order parameter class.
* Updated AngularSeparation class.
* Rewrote Voronoi implementation to leverage voro++.
* Implemented Voronoi bond weighting to enable Minkowski structure metrics.
* Refactored methods in Box and PeriodicBuffer for v2.0.
* Added checks to C++ for 2D boxes where required.
* Refactored cluster module.
* Standardized vector directionality in computes.
* NeighborQuery support to ClusterProperties, GaussianDensity, Voronoi, PeriodicBuffer, Interface.
* Standardized APIs for order parameters.
* Added radius of gyration to ClusterProperties.
* Improved Voronoi plotting code.
* Corrected number of points/query points in LocalDensity.
* Made PeriodicBuffer inherit from _Compute.
* Removed cudacpu and HOOMDMath includes.
* Added plotting functionality for Box and NeighborQuery objects.
* Added support for reading system data directly from MDAnalysis, garnett, gsd, HOOMD-blue, and OVITO.
* Revised tutorials and documentation on data inputs.
* Updated MSD to perform accumulation with ``compute(..., reset=False)``.
* Added test PyPI support to continuous integration.
* Added continuous integration to freud-examples.
* Implemented periodic center of mass computations in C++.
* Revised docs about query modes.
* Implemented smarter heuristics in Voronoi for voro++ block sizes, resulting in significant performance gains for large systems.
* Corrected calculation of neighbor distances in the Voronoi NeighborList.
* Added finite tolerance to ensure stability of 2D Voronoi NeighborList computations.
* Improved stability of Histogram bin calculations.
* Improved error handling of Cubatic input parameters.
* Added 2D Minkowski Structure Metrics to Hexatic, enabled by using ``weighted=True`` along with a Voronoi NeighborList.
* Worked with Tommy Waltmann to add the SphereVoxelization feature.
* Fixed GaussianDensity normalization in 2D systems.
* Prevented GaussianDensity from computing 3D systems after it has computed 2D systems.
* Contributed code, design, and testing for ``DiffractionPattern`` class.
* Fixed ``Hexatic`` order parameter (unweighted) to normalize by number of neighbors instead of the symmetry order k.
* Added ``num_query_points`` and ``num_points`` attributes to NeighborList class.
* Added scikit-build support for Windows.
* Fixed 2D image calculations.
* Optimized NeighborList ``filter`` method.

Eric Harper, University of Michigan - **Former lead developer**

* Added TBB parallelism.
* Wrote PMFT module.
* Added NearestNeighbors (since removed).
* Wrote RDF.
* Added bonding module (since removed).
* Added cubatic order parameter.
* Added hexatic order parameter.
* Added Pairing2D (since removed).
* Created common array conversion logic.

Joshua A. Anderson, University of Michigan - **Creator and former lead developer**

* Initial design and implementation.
* Wrote LinkCell and IteratorLinkCell.
* Wrote GaussianDensity, LocalDensity.
* Added parallel module.
* Added indexing modules (since removed).
* Wrote Cluster and ClusterProperties modules.

Matthew Spellings - **Former lead developer**

* Added generic neighbor list.
* Enabled neighbor list usage across freud modules.
* Added correlation functions.
* Added LocalDescriptors class.
* Added interface module.

Erin Teich

* Wrote environment matching (MatchEnv) class.
* Wrote BondOrder class (with Julia Dshemuchadse).
* Wrote AngularSeparation class (with Andrew Karas).
* Contributed to LocalQl development.
* Wrote LocalBondProjection class.

M. Eric Irrgang

* Authored kspace module (since removed).
* Fixed numerous bugs.
* Contributed to freud.shape (since removed).

Chrisy Du

* Authored Steinhardt order parameter classes.
* Fixed support for triclinic boxes.

Antonio Osorio

* Developed TrajectoryXML class.
* Various bug fixes.
* OpenMP support.

Richmond Newman

* Developed the freud box.
* Solid liquid order parameter.

Carl Simon Adorf

* Developed the Python box module.

Jens Glaser

* Wrote kspace front-end (since removed).
* Modified kspace module (since removed).
* Wrote Nematic order parameter class.

Benjamin Schultz

* Wrote Voronoi class.
* Fix normalization in GaussianDensity.
* Bug fixes in shape module (since removed).

Bryan VanSaders

* Make Cython catch C++ exceptions.
* Add shiftvec option to PMFT.

Ryan Marson

* Various GaussianDensity bugfixes.

Yina Geng

* Co-wrote Voronoi neighbor list module.
* Add properties for accessing class members.

Carolyn Phillips

* Initial design and implementation.
* Package name.

Ben Swerdlow

* Documentation and installation improvements.

James Antonaglia

* Added number of neighbors as an argument to HexOrderParameter.
* Bugfixes.
* Analysis of deprecated kspace module.

Mayank Agrawal

* Co-wrote Voronoi neighbor list module.

William Zygmunt

* Helped with Boost removal.

Greg van Anders

* Bugfixes for CMake and SSE2 installation instructions.

James Proctor

* Cythonization of the cluster module.

Rose Cersonsky

* Enabled TBB-parallelism in density module.
* Fixed how C++ arrays were pulled into Cython.

Wenbo Shen

* Translational order parameter.

Andrew Karas

* Angular separation.
* Wrote reference implementation for rotational autocorrelation.

Paul Dodd

* Fixed CorrelationFunction namespace, added ComputeOCF class for TBB parallelization.

Tim Moore

* Added optional rmin argument to density.RDF.
* Enabled NeighborList indexing.

Alex Dutton

* BiMap class for MatchEnv.

Matthew Palathingal

* Replaced use of boost shared arrays with shared ptr in Cython.
* Helped incorporate BiMap class into MatchEnv.

Kelly Wang

* Enabled NeighborList indexing.
* Added methods ``compute_distances`` and ``compute_all_distances`` to Box.
* Added method ``crop`` to Box.
* Added 2D Box tests for ``get_image`` and ``contains``.

Yezhi Jin

* Added support for 2D arrays in the Python interface to Box functions.
* Rewrote Voronoi implementation to leverage voro++.
* Implemented Voronoi bond weighting to enable Minkowski structure metrics.
* Contributed code, design, and testing for ``DiffractionPattern`` class.

Brandon Butler

* Rewrote Steinhardt order parameter.

Jin Soo Ihm

* Added benchmarks.
* Contributed to NeighborQuery classes.
* Refactored C++ to perform neighbor queries on-the-fly.
* Added plotting functions to analysis classes.
* Wrote RawPoints class.
* Created Compute parent class with decorators to ensure properties have been computed.
* Updated common array conversion logic.
* Added many validation tests.

Mike Henry

* Fixed syntax in freud-examples notebooks for v2.0.
* Updated documentation links

Michael Stryk

* Added short examples into Cluster, Density, Environment, and Order Modules.

Tommy Waltmann

* Worked with Bradley Dice to add the SphereVoxelization feature.
* Contributed code, design, and testing for ``DiffractionPattern`` class.

Maya Martirossyan

* Added test for Steinhardt for particles without neighbors.

Pavel Buslaev

* Added ``values`` argument to compute method of ``GaussianDensity`` class.

Source code
-----------

.. highlight:: none

Eigen (http://eigen.tuxfamily.org) is included as a git submodule in freud.
Eigen is made available under the Mozilla Public License v2.0
(http://mozilla.org/MPL/2.0/). Its linear algebra routines are used for
various tasks including the computation of eigenvalues and eigenvectors.

fsph (https://github.com/glotzerlab/fsph) is included as a git submodule in
freud. It is used for the calculation of spherical harmonics. fsph is made
available under the MIT license::

    Copyright (c) 2016 The Regents of the University of Michigan

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

HOOMD-blue (https://github.com/glotzerlab/hoomd-blue) is the original source of
some algorithms and tools for vector math implemented in freud. HOOMD-blue is
made available under the BSD 3-Clause license::

	BSD 3-Clause License for HOOMD-blue

	Copyright (c) 2009-2019 The Regents of the University of Michigan All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are met:

	1. Redistributions of source code must retain the above copyright notice,
	   this list of conditions and the following disclaimer.

	2. Redistributions in binary form must reproduce the above copyright notice,
	   this list of conditions and the following disclaimer in the documentation
	   and/or other materials provided with the distribution.

	3. Neither the name of the copyright holder nor the names of its contributors
	   may be used to endorse or promote products derived from this software without
	   specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
	ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

voro++ (https://github.com/chr1shr/voro) is included as a git submodule in
freud. It is used for computing Voronoi diagrams. voro++ is made available
under the following license::

    Voro++ Copyright (c) 2008, The Regents of the University of California, through
    Lawrence Berkeley National Laboratory (subject to receipt of any required
    approvals from the U.S. Dept. of Energy). All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    (1) Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

    (2) Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

    (3) Neither the name of the University of California, Lawrence Berkeley
    National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
    ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    You are under no obligation whatsoever to provide any bug fixes, patches, or
    upgrades to the features, functionality or performance of the source code
    ("Enhancements") to anyone; however, if you choose to make your Enhancements
    available either publicly, or directly to Lawrence Berkeley National
    Laboratory, without imposing a separate written license agreement for such
    Enhancements, then you hereby grant the following license: a non-exclusive,
    royalty-free perpetual license to install, use, modify, prepare derivative
    works, incorporate into other computer software, distribute, and sublicense
    such enhancements or derivative works thereof, in binary and source code form.
