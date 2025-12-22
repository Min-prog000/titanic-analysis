# titanic-analysis

![python_badge](https://img.shields.io/badge/python-3.13.6-blue?logo=python&logoColor=white)
![uv_badge](https://img.shields.io/badge/uv-0.8.9-g?logo=uv&logoColor=white)
![uv_badge](https://img.shields.io/badge/ruff-0.12.8-g?logo=ruff&logoColor=white)

## Outline

- [1. Overview](#1-overview)
- [2. Layout](#2-layout)
- [3. Install](#3-install)
- [4. Usage](#4-usage)
- [5. Contributing](#5-contributing)
- [6. Dataset](#6-dataset)
- [7. License](#7-license)
- [8. Changelog](#8-changelog)

## [1. Overview](#1-overview)

This repository is created for titanic dataset analysis and development AI model.

### Goal

- Target accuracy: **90% or higher**
- Candidate models: **Logistic regression**, **Random forest**, **Gradient boosting**, and **Multi-Layer Perceptron**.

### Features

- Currently under development
- Support multiple execution mode via arguments parsing

## [2. Layout](#2-layout)

```text
titanic-analysis/
  ├── notebook/
  │     └ main_notebook.ipynb
  ├── src/
  │     └── titanic_analysis/
  │             ├── domain/
  │             ├── framework/
  │             ├── infrastructure/
  │             ├── usecase/
  │             └── main.py
  ├── .gitattributes
  ├── .gitignore
  ├── CHANGELOG.md
  ├── CONTRIBUTING.md
  ├── LICENSE
  ├── pyproject.toml
  ├── README.md
  └── requirements.txt
```

## [3. Install](#3-install)

- Python: 3.13.6
- Project management: uv
- Linter / Formatter: Ruff

### [3.1 Install uv]

In windows PowerShell:

```shell
PS> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

In macOS/Linux:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Details of install provided in [Installing uv](https://docs.astral.sh/uv/getting-started/installation/)

### [3.2 Initialize uv project]

In your folder cloned this repository:

```shell
uv init
```

### [3.3 Create `.venv`]

In root folder of initialized project:

```shell
uv venv .venv
```

### [3.4 Activate virtual environment]

In root folder, activate virtual environment with `.venv\Scripts\activate.bat`.

In Windows:

```shell
.venv\Scripts\activate
```

In macOS/Linux:

```shell
source .venv/bin/activate
```

## [4. Usage](#4-usage)

### CLI

- Sample 1 - Analysis

```shell
uv run titanic-analysis
# or
uv run titanic-analysis -m 0
```

- Sample 2 - Training

(Coming soon)

## [5. Contributing](#5-contributing)

Welcome presentation to improvement codes and structure.
See `CONTRIBUTING.md` about PR or commit (reviewer is the owner basically)

## [6. Dataset](#6-dataset)

This project uses the Titanic dataset provided by Kaggle for the
[Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic-dataset/data) competition.
The dataset is subject to Kaggle's competition rules and is not redistributed
within this repository. Please download it directly from Kaggle.

## [7. License](#7-license)

This project is licensed under the Apache License 2.0

## [8. CHANGELOG](#8-changelog)

See update history in: [CHANGELOG.md](https://github.com/Min-prog000/titanic-analysis/blob/main/CHANGELOG.md)
