import sys
import os
from skbuild import setup as skbuild_setup

version = '2.4.1'

# Read README for PyPI, fallback to short description if it fails.
desc = 'Powerful, efficient trajectory analysis in scientific Python.'
try:
    readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'README.rst')
    with open(readme_file) as f:
        readme = f.read()
except ImportError:
    readme = desc


def setup(*args, **kwargs):
    """This wrapper exists to force the option --build-type=ReleaseWithDocs.

    Neither Release nor RelWithDebInfo will work, due to hard-coded options in
    scikit-build's UseCython.cmake that disable docstrings. The choice of
    ReleaseWithDocs is arbitrary, as a string that won't overlap with any build
    type handled in UseCython.cmake. See this issue for details:
    https://github.com/scikit-build/scikit-build/issues/518
    """
    BUILD_TYPE = '--build-type=ReleaseWithDocs'
    for index, arg in enumerate(sys.argv):
        if arg == '--':
            # Insert at the end of the options that go to scikit-build
            break
        elif arg.startswith('--build-type'):
            # Don't override user-specified build type
            index = False
            break
    else:
        # Insert at the end of the provided arguments
        index = len(sys.argv)
    if index:
        sys.argv.insert(index, BUILD_TYPE)
    skbuild_setup(*args, **kwargs)


setup(
    name='freud-analysis',
    version=version,
    packages=['freud'],
    description=desc,
    long_description=readme,
    long_description_content_type='text/x-rst',
    keywords=('simulation analysis molecular dynamics soft matter '
              'particle system computational physics'),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: C++",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    zip_safe=False,
    maintainer='freud Developers',
    maintainer_email='freud-developers@umich.edu',
    # See documentation credits for current and former lead developers
    author='Vyas Ramasubramani et al.',
    author_email='vramasub@umich.edu',
    url='https://github.com/glotzerlab/freud',
    download_url='https://pypi.org/project/freud-analysis/',
    project_urls={
        'Homepage': 'https://github.com/glotzerlab/freud',
        'Documentation': 'https://freud.readthedocs.io/',
        'Source Code': 'https://github.com/glotzerlab/freud',
        'Issue Tracker': 'https://github.com/glotzerlab/freud/issues',
    },
    python_requires='>=3.6',
    install_requires=[
        'cython>=0.29',
        'numpy>=1.14',
        'rowan>=1.2',
        'scipy>=1.1',
    ],
    tests_require=[
        'gsd>=2.0',
        'garnett>=0.7.1',
        'matplotlib>=3.0',
        'MDAnalysis>=0.20.1',
        'sympy>=1.0',
    ])
