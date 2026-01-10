# titanic-analysis

![python_badge](https://img.shields.io/badge/python-3.13.6-blue?logo=python&logoColor=white)
![uv_badge](https://img.shields.io/badge/uv-0.8.9-g?logo=uv&logoColor=white)
![uv_badge](https://img.shields.io/badge/ruff-0.12.8-g?logo=ruff&logoColor=white)

## Outline

- [1. Overview](#1-overview)
- [2. Layout](#2-layout)
- [3. Dataset](#3-dataset)
- [4. Install](#4-install)
- [5. Usage](#5-usage)
- [6. Contributing](#6-contributing)
- [7. License](#7-license)
- [8. Changelog](#8-changelog)

## [1. Overview](#1-overview)

This repository is dedicated to analyzing the Titanic dataset and developing AI models based on it.

### 1.1 Goal

- Target accuracy: **90% or higher**
- Candidate models: **Logistic regression**, **Random forest**, **Gradient boosting**, and **Multi-Layer Perceptron**.

### 1.2 Features

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

## [3. Dataset](#3-dataset)

This project uses the Titanic dataset provided by Kaggle for the
[Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data) competition.
The dataset is subject to Kaggle's competition rules and is not redistributed within this repository.
Please download it directly from Kaggle.

```text
titanic-analysis/
  ├── data/  # <- Please add train.csv and test.csv in this position
  │     └── titanic/
  │             ├── train.csv
  │             └── test.csv
  ├── notebook/
  │     └── main_notebook.ipynb
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

## [4. Install](#4-install)

- Python: 3.13.6
- Project management: uv
- Linter / Formatter: Ruff
- Build system: uv-build

### 4.1 Install uv

On Windows PowerShell:

```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

On macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For more details, see the official guide: [Installing uv](https://docs.astral.sh/uv/getting-started/installation/)

### 4.2 Create `.venv`

In the root folder of the initialized project, run:

```shell
uv venv .venv
```

### 4.3 Activate virtual environment

In the root folder, activate the virtual environment using `.venv\Scripts\activate.bat`.

On Windows:

```shell
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 4.4 Install packages with `requirements.txt`

In the virtual environment, run command:

```shell
uv pip install -r requirements.txt
```

### 4.5 Set execution environment for `main_notebook.ipynb`

Open `main_notebook.ipynb`, set `titanic-analysis` to kernel.

## [5. Usage](#5-usage)

### CLI

- Sample 1 - Analysis

```shell
uv run titanic-analysis
# or
uv run titanic-analysis -m 0
```

- Sample 2 - Training

(Coming soon)

### Jupyter Notebook

1. Execute from `Libraries Importing` to `Classes`.
2. Run the cells under **Data analysis** to explore and preprocess the dataset.
3. Proceed to **Learning** to train the machine learning models.
  
`Data preparation` and `Getting column names`

Executable methods:

- Logistic regression
- Random forest

## [6. Contributing](#6-contributing)

We welcome contributions that improve the code and project structure.  
For details on pull requests and commits, please refer to `CONTRIBUTING.md`  
(the project owner is typically the reviewer).

## [7. License](#7-license)

This project is licensed under the Apache License 2.0

## [8. CHANGELOG](#8-changelog)

See update history in: [CHANGELOG.md](https://github.com/Min-prog000/titanic-analysis/blob/main/CHANGELOG.md)
