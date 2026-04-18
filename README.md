# npest: Nonparametric Estimation and Testing in Python

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**npest** is a lightweight, pure Python library for intuitive nonparametric inference. It provides:

- **Kernel Density Estimation** (Gaussian, Epanechnikov, Triangular, Box)
- **Histograms** with automatic bin selection (Sturges, Freedman–Diaconis, Scott, Bayesian blocks)
- **Bayesian bootstrap** for credible intervals
- **Nonparametric hypothesis tests**: goodness-of-fit, two-sample, correlation (Spearman, Kendall, Hoeffding), and custom bootstrap tests.

The library follows a consistent scikit-learn–inspired API: `fit()`, `pdf()`, `plot()`.

## Installation
You can install the latest version directly from the GitHub repository:
```bash
pip install git+https://github.com/23ssrivast/npest.git
