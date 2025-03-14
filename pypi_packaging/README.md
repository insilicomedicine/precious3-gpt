# PyPI Packaging for P3GPT

This directory contains files necessary for publishing the P3GPT package to PyPI.

## Publishing to PyPI

Follow these steps to publish a new version of the package:

1. Update the version number in both `pyproject.toml` and `setup.py`

2. Build the distribution packages:
   ```bash
   python -m build
   ```

3. Upload the packages to PyPI:
   ```bash
   python -m twine upload dist/*
   ```

4. To upload to Test PyPI first (recommended for testing):
   ```bash
   python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
   ```

## Files

- `pyproject.toml`: Modern Python packaging configuration
- `setup.py`: Traditional setup file for backward compatibility
- `MANIFEST.in`: Specifies additional files to include in the distribution

## Versioning

Follow [Semantic Versioning](https://semver.org/) for version numbers:
- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions
- PATCH version for backwards-compatible bug fixes

## Dependencies

Dependencies are specified in both `pyproject.toml` and `requirements.txt`. Make sure to keep them in sync when updating.
