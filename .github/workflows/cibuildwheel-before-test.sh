#!/bin/bash
if [ -z $1 ]; then
    echo "A package directory must be provided as the first argument."
    exit 1
fi

PACKAGE_DIR=$1
PYTHON_VERSION=$(python --version 2>&1)

if [[ "${PYTHON_VERSION}" == *"Python 3.6."* ]]; then
  # Python 3.6 is only supported with oldest requirements
  pip install -U -r "${PACKAGE_DIR}/.circleci/ci-oldest-reqs.txt" --progress-bar=off
else
  pip install -U -r "${PACKAGE_DIR}/requirements/requirements-test.txt" --progress-bar=off
fi

# Allow parallel tests to speed up CI
pip install -U pytest-xdist --progress-bar=off
