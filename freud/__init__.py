# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from . import box
from . import cluster
from . import data
from . import density
from . import diffraction
from . import environment
from . import interface
from . import locality
from . import msd
from . import order
from . import parallel
from . import pmft

from .box import Box
from .locality import AABBQuery, LinkCell, NeighborList
from .parallel import get_num_threads, set_num_threads, NumThreads

# Override TBB's default autoselection. This is necessary because once the
# automatic selection runs, the user cannot change it.
set_num_threads(0)

__version__ = '2.4.1'

__all__ = [
    '__version__',
    'box',
    'cluster',
    'data',
    'density',
    'diffraction',
    'environment',
    'interface',
    'locality',
    'msd',
    'order',
    'parallel',
    'pmft',
    'Box',
    'AABBQuery',
    'LinkCell',
    'NeighborList',
    'get_num_threads',
    'set_num_threads',
    'NumThreads',
]

__citation__ = """@article{freud2020,
    title = {freud: A Software Suite for High Throughput
             Analysis of Particle Simulation Data},
    author = {Vyas Ramasubramani and
              Bradley D. Dice and
              Eric S. Harper and
              Matthew P. Spellings and
              Joshua A. Anderson and
              Sharon C. Glotzer},
    journal = {Computer Physics Communications},
    volume = {254},
    pages = {107275},
    year = {2020},
    issn = {0010-4655},
    doi = {https://doi.org/10.1016/j.cpc.2020.107275},
    url = {http://www.sciencedirect.com/science/article/pii/S0010465520300916},
    keywords = {Simulation analysis, Molecular dynamics, Monte Carlo,
                Computational materials science},
}"""
