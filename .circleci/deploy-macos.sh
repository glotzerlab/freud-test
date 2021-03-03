#!/bin/bash

set -e

# PYPI_USERNAME - (Required) Username for the publisher's account on PyPI
# PYPI_PASSWORD - (Required, Secret) Password for the publisher's account on PyPI

cat << EOF > ~/.pypirc
[distutils]
index-servers=
    pypi
    testpypi

[pypi]
username: ${PYPI_USERNAME}
password: ${PYPI_PASSWORD}

[testpypi]
repository: https://test.pypi.org/legacy/
username: ${PYPI_USERNAME}
password: ${PYPI_PASSWORD}
EOF

if [ -z $1 ]; then
    echo "A repository (\"pypi\" or \"testpypi\") must be provided as the first argument."
    exit 1
fi

export MACOSX_DEPLOYMENT_TARGET=10.12
# When homebrew installs pyenv it tries to update the Python patch versions to
# the latest. Those updates automatically delete the older versions (e.g.
# 3.8.5->3.8.6 leads to the 3.8.5 folder being deleted), but then homebrew
# tries to delete them again and causes errors. To avoid this issue, we update
# manually here for the current main version of Python (required by pyenv).
brew upgrade python@3.8
brew install pyenv
eval "$(pyenv init -)"
# Check supported versions with pyenv install --list
PY_VERSIONS=(3.6.12 3.7.9 3.8.6 3.9.0)

# Build TBB
git clone https://github.com/intel/tbb.git
cd tbb
make
BUILD_DIR=$(find build -name mac*release)
cd ${BUILD_DIR}
source tbbvars.sh
# Force the TBB path to use an absolute path to itself for others to find.
install_name_tool -id "${PWD}/libtbb.dylib" libtbb.dylib
cd ~/

# Build wheels
for VERSION in ${PY_VERSIONS[@]}; do
  echo "Building for Python ${VERSION}"
  pyenv install ${VERSION}
  pyenv global ${VERSION}

  pip install --upgrade pip
  pip install cython scikit-build cmake --ignore-installed -q --progress-bar=off
  rm -rf numpy-1.14.6
  curl -sSLO https://github.com/numpy/numpy/archive/v1.14.6.tar.gz
  tar -xzf v1.14.6.tar.gz
  cd numpy-1.14.6
  rm -f numpy/random/mtrand/mtrand.c
  rm -f PKG-INFO
  pip install . --no-deps --ignore-installed -v -q --progress-bar=off

  pip install wheel delocate --progress-bar=off
  pip wheel ~/ci/freud/ -w ~/wheelhouse/ --no-deps --no-build-isolation --no-use-pep517
done

# Update RPath for wheels
for whl in ~/wheelhouse/freud*.whl; do
  delocate-wheel "$whl" -w ~/ci/freud/wheelhouse/
done

# Install from and test all wheels
for VERSION in ${PY_VERSIONS[@]}; do
  echo "Testing for Python ${VERSION}"
  pyenv global ${VERSION}

  pip install freud_analysis --no-deps --no-index -f ~/ci/freud/wheelhouse
  # Don't install MDAnalysis and skip the relevant tests.
  cat ~/ci/freud/requirements-testing.txt | grep -v MDAnalysis | xargs -n 1 pip install -U --progress-bar=off
  cd ~/ci/freud/tests
  python -m unittest discover . -v
done

pip install --user twine

python -m twine upload --skip-existing --repository $1 ~/ci/freud/wheelhouse/*
