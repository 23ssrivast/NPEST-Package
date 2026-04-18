# npest: Nonparametric Estimation and Testing in Python

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**npest** is a lightweight, pure Python library for intuitive nonparametric inference. This library is useful for estimation purposes for Non Parametric datasets. It provides:

- **Kernel Density Estimation** (Gaussian, Epanechnikov, Triangular, Box)
- **Histograms** with automatic bin selection (Sturges, Freedman–Diaconis, Scott, Bayesian blocks)
- **Bayesian bootstrap** for credible intervals
- **Nonparametric hypothesis tests**: goodness-of-fit test, two-sample, correlation (Spearman, Kendall, Hoeffding), and custom bootstrap tests.

The library follows a consistent scikit-learn–inspired API: `fit()`, `pdf()`, `plot()`.

## Installation
Option 1:
You can install the latest version directly from the GitHub repository using command prompt:
```bash
curl -O https://raw.githubusercontent.com/23ssrivast/NPEST-Package/main/npest.py
```
Option 2:
Use this directly in python
```bash
import urllib.request
import os

# The direct link to the raw Python file on GitHub
url = "https://raw.githubusercontent.com/23ssrivast/NPEST-Package/main/npest.py"

# The name you want to save it as locally
filename = "npest.py"

print(f"Downloading {filename} from GitHub...")

try:
    # This downloads the file and saves it
    urllib.request.urlretrieve(url, filename)
    print(f"Success! '{filename}' has been saved to: {os.getcwd()}")
    print("You can now type 'import npest' to use it.")
except Exception as e:
    print(f"An error occurred: {e}")
```
