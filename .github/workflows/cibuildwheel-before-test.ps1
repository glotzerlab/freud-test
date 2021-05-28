$PACKAGE_DIR = $args[0]
$PYTHON_VERSION = $(python --version)

If ("${PYTHON_VERSION}" -Match "Python 3\.6\.") {
    # Python 3.6 is only supported with oldest requirements
    pip install -U -r ${PACKAGE_DIR}/.circleci/ci-oldest-reqs.txt --progress-bar=off
} Else {
    pip install -U -r ${PACKAGE_DIR}/requirements/requirements-test.txt --progress-bar=off
}

# Allow parallel tests to speed up CI
#pip install -U pytest-xdist --progress-bar=off
