# titanic-analysis

![python_badge](https://img.shields.io/badge/python-3.13.6-blue?logo=python&logoColor=white)
![uv_badge](https://img.shields.io/badge/uv-0.8.9-g?logo=uv&logoColor=white)
![uv_badge](https://img.shields.io/badge/ruff-0.12.8-g?logo=ruff&logoColor=white)

## [1. Overview](#1-overview)

This repository is created for titanic dataset analysis and development AI model.

### Goal

- Accuracy: More than **90** percentages
- Achievement methods: One of **Logistic regression**, **Random forest**, **Gradient boosting**, and **Multi-perceptron**.

### Features

- Now in development
- Choose execution mode with arguments parser

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

- Project management - uv
- Python version - 3.13.6
- Linter / Formatter - Ruff

## [4. Usage](#4-usage)

### CLI

- Sample 1 - Analysis

```shell
uv run titanic-analysis
# or
uv run titanic-analysis -m 0
```

- Sample 2 - Training (Future available)

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

Update history: [CHANGELOG.md](https://github.com/Min-prog000/titanic-analysis/blob/main/CHANGELOG.md)
