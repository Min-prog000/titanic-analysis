# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

(None)

### Changed

(None)

### Fixed

(None)

### Model

(None)

### Dataset

(None)

### API

(None)

### Removed

(None)

### Duplicated

(None)

## [0.2.2] - 2026-01-02

### Fixed

- Fix codes because of verifying successful execution of `main_notebook.ipynb` ([#29](https://github.com/Min-prog000/titanic-analysis/pull/29), fixes [#26](https://github.com/Min-prog000/titanic-analysis/issues/26))

## [0.2.1] - 2026-01-01

### Fixed

- Fix codes because of verifying successful execution of `main.py` ([#27](https://github.com/Min-prog000/titanic-analysis/pull/27), fixes [#25](https://github.com/Min-prog000/titanic-analysis/issues/25))

## [0.2.0] - 2026-01-01

### Added

- Migrated `main.ipynb` from private repository ([#24](https://github.com/Min-prog000/titanic-analysis/pull/24), fixes [#21](https://github.com/Min-prog000/titanic-analysis/issues/21))

### Changed

- Rename migrated `main.ipynb` to `main_notebook.ipynb` ([#24](https://github.com/Min-prog000/titanic-analysis/pull/24), fixes [#21](https://github.com/Min-prog000/titanic-analysis/issues/21))

## [0.1.0] - 2025-12-31

### Added

- Added `CHANGELOG.md` ([#6](https://github.com/Min-prog000/titanic-analysis/pull/6), fixes [#5](https://github.com/Min-prog000/titanic-analysis/issues/5))
- Added `CONTRIBUTING.md` ([#12](https://github.com/Min-prog000/titanic-analysis/pull/12), fixes [#11](https://github.com/Min-prog000/titanic-analysis/issues/11))
- Setup `uv` project ([#14](https://github.com/Min-prog000/titanic-analysis/pull/14), fixes [#7](https://github.com/Min-prog000/titanic-analysis/issues/7))
  - Create `uv` project using `uv init` ([#14](https://github.com/Min-prog000/titanic-analysis/pull/14), fixes [#8](https://github.com/Min-prog000/titanic-analysis/issues/8))
  - Added `.venv` in local environment ([#14](https://github.com/Min-prog000/titanic-analysis/pull/14), fixes [#9](https://github.com/Min-prog000/titanic-analysis/issues/9))
- Introduced custom package structure under `src/titanic_analysis`: ([#23](https://github.com/Min-prog000/titanic-analysis/pull/23), fixes [#20](https://github.com/Min-prog000/titanic-analysis/issues/20))
  - **domain**: core business logic and entities
  - **framework**: application framework and external interfaces
  - **infrastructure**: data access and system-level implementations
  - **usecase**: application-specific use cases and workflows
- Added `requirements.txt` ([#33](https://github.com/Min-prog000/titanic-analysis/pull/33), fixes [#31](https://github.com/Min-prog000/titanic-analysis/issues/31))

### Changed

- Updated documentations related `LICENSE`
  - Updated `LICENSE` with copyright notice (2025 Minplu) ([#3](https://github.com/Min-prog000/titanic-analysis/pull/3), fixes [#1](https://github.com/Min-prog000/titanic-analysis/issues/1))
  - Added "License" section to `README.md` referencing `LICENSE` ([#4](https://github.com/Min-prog000/titanic-analysis/pull/4), fixes [#2](https://github.com/Min-prog000/titanic-analysis/issues/2))
- Setup `uv` project ([#14](https://github.com/Min-prog000/titanic-analysis/pull/14), fixes [#7](https://github.com/Min-prog000/titanic-analysis/issues/7))
  - Updated `.gitignore` to validate to `.vscode`, `uv.lock` and `.python-version` ([#14](https://github.com/Min-prog000/titanic-analysis/pull/14), fixes [#13](https://github.com/Min-prog000/titanic-analysis/issues/13))
  - Updated libraries dependencies to `pyproject.toml` ([#14](https://github.com/Min-prog000/titanic-analysis/pull/14), fixes [#10](https://github.com/Min-prog000/titanic-analysis/issues/10))
    - `numpy`
    - `pandas`
    - `scikit-learn`
- Updated `pyproject.toml` to add linter setting with `Ruff` ([#18](https://github.com/Min-prog000/titanic-analysis/pull/18), fixes [#17](https://github.com/Min-prog000/titanic-analysis/issues/17))
- Move `main.py` to `src/titanic_analysis` ([#23](https://github.com/Min-prog000/titanic-analysis/pull/23), fixes [#20](https://github.com/Min-prog000/titanic-analysis/issues/20))
- Adjust `main.py` to execution using `uv run` ([#23](https://github.com/Min-prog000/titanic-analysis/pull/23), fixes [#20](https://github.com/Min-prog000/titanic-analysis/issues/20))
- Revised `README.md` with Kaggle dataset usage instructions ([#23](https://github.com/Min-prog000/titanic-analysis/pull/23), fixes [#20](https://github.com/Min-prog000/titanic-analysis/issues/20))
- Updated `pyproject.toml` to add dependencies ([#23](https://github.com/Min-prog000/titanic-analysis/pull/23), fixes [#20](https://github.com/Min-prog000/titanic-analysis/issues/20))
  - `pyyaml`
  - `pydantic`
- Updated `README.md` to revise outlines and dataset URL, and add execution and installation instructions ([#33](https://github.com/Min-prog000/titanic-analysis/pull/33), fixes [#30](https://github.com/Min-prog000/titanic-analysis/issues/30))
