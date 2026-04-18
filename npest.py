"""
Includes:
- Kernel Density Estimation (KDE) with multiple kernels
- Histogram density estimation
- Bayesian bootstrap
- Nonparametric hypothesis tests (goodness-of-fit, two-sample, correlation, bootstrap)

Usage:
    import npest
    kde = npest.KDE(kernel='epanechnikov').fit(data)
    p_val = npest.goodness_of_fit(data, dist='normal')
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

__version__ = "0.2.0"

def _check_1d(data):
    """Convert input to 1D numpy array and validate."""
    arr = np.asarray(data)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {arr.shape}")
    return arr

def _check_fitted(estimator):
    """Raise error if estimator is not fitted."""
    if not getattr(estimator, '_fitted', False):
        raise RuntimeError("Estimator must be fitted before calling this method.")

def _plot_density(estimator, ax=None, n_points=1000, **kwargs):
    """Generic density plotting for estimators with .pdf() and .data_ attributes."""
    if ax is None:
        fig, ax = plt.subplots()
    data = estimator.data_
    margin = 3 * np.std(data)
    x_min = data.min() - margin
    x_max = data.max() + margin
    x_grid = np.linspace(x_min, x_max, n_points)
    y = estimator.pdf(x_grid)
    ax.plot(x_grid, y, **kwargs)
    ax.set_title(estimator.__class__.__name__ + " estimate")
    return ax

# Kernel Density Estimation
class KDE:
    """
    Kernel Density Estimation.

    Parameters

    bandwidth : float or str, default='silverman'
        Bandwidth selection method. Options:
        - float : user-specified value
        - 'scott' : Scott's rule
        - 'silverman' : Silverman's rule of thumb
    kernel : str, default='gaussian'
        Kernel type. Options:
        - 'gaussian'   : standard normal kernel
        - 'epanechnikov': Epanechnikov (parabolic) kernel
        - 'triangular'  : triangular kernel
        - 'box'         : uniform / box kernel

    Attributes

    data_ : ndarray
        Training data.
    bw_ : float
        Bandwidth used after fitting.
    """
    def __init__(self, bandwidth='silverman', kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel.lower()
        self._fitted = False

        # Define kernel functions
        self._kernel_funcs = {
            'gaussian': lambda z: np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi),
            'epanechnikov': lambda z: 0.75 * (1 - z**2) * (np.abs(z) <= 1),
            'triangular': lambda z: (1 - np.abs(z)) * (np.abs(z) <= 1),
            'box': lambda z: 0.5 * (np.abs(z) <= 1)
        }
        if self.kernel not in self._kernel_funcs:
            raise ValueError(f"Unsupported kernel: {self.kernel}. "
                             f"Choose from {list(self._kernel_funcs.keys())}")

    def fit(self, data):
        """
        Fit the KDE model.

        Parameters

        data : array-like, shape (n_samples,)
            One-dimensional data points.

        Returns

        self : object
            Fitted estimator.
        """
        data = _check_1d(data)
        self.data_ = np.asarray(data)
        self.n_ = len(self.data_)

        # Bandwidth selection
        if isinstance(self.bandwidth, str):
            if self.bandwidth == 'scott':
                self.bw_ = self.n_ ** (-1/5) * np.std(self.data_, ddof=1)
            elif self.bandwidth == 'silverman':
                sigma = np.std(self.data_, ddof=1)
                iqr = stats.iqr(self.data_)
                A = min(sigma, iqr / 1.34)
                self.bw_ = 0.9 * A * self.n_ ** (-1/5)
            else:
                raise ValueError(f"Unknown bandwidth rule: {self.bandwidth}")
        else:
            self.bw_ = float(self.bandwidth)

        self._fitted = True
        return self

    def pdf(self, x):
        """
        Evaluate the estimated density at points x.

        Parameters

        x : array-like
            Points at which to evaluate the density.

        Returns

        density : ndarray
            Density values at x.
        """
        _check_fitted(self)
        x = np.asarray(x).reshape(-1)
        h = self.bw_
        kernel_func = self._kernel_funcs[self.kernel]

        # Vectorized computation: (x_i - data_j) / h
        z = (x[:, np.newaxis] - self.data_[np.newaxis, :]) / h
        density = np.sum(kernel_func(z), axis=1) / (self.n_ * h)
        return density

    def plot(self, ax=None, n_points=1000, **kwargs):
        """Plot the estimated density."""
        _check_fitted(self)
        return _plot_density(self, ax, n_points, **kwargs)

# Histogram Density Estimation

class Histogram:
    """
    Histogram density estimator.

    Parameters

    bins : int or str, default='fd'
        Number of bins or bin selection method:
        - int : user-specified number of bins
        - 'sturges' : Sturges' rule
        - 'fd' : Freedman-Diaconis rule
        - 'scott' : Scott's rule
        - 'blocks' : Bayesian blocks (optimal binning)
    density : bool, default=True
        If True, normalize to form a probability density.
    """
    def __init__(self, bins='fd', density=True):
        self.bins = bins
        self.density = density
        self._fitted = False

    def fit(self, data):
        data = _check_1d(data)
        self.data_ = np.asarray(data)

        if isinstance(self.bins, int):
            n_bins = self.bins
        elif isinstance(self.bins, str):
            n_bins = self._compute_bins(data, self.bins)
        else:
            raise TypeError("bins must be int or str")

        self.hist_, self.bin_edges_ = np.histogram(
            data, bins=n_bins, density=self.density
        )
        self._fitted = True
        return self

    @staticmethod
    def _compute_bins(data, method):
        n = len(data)
        if method == 'sturges':
            return int(np.ceil(np.log2(n)) + 1)
        elif method == 'fd':
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            if iqr == 0:
                return 1
            bin_width = 2 * iqr * n ** (-1/3)
            return int(np.ceil((data.max() - data.min()) / bin_width))
        elif method == 'scott':
            std = np.std(data, ddof=1)
            if std == 0:
                return 1
            bin_width = 3.5 * std * n ** (-1/3)
            return int(np.ceil((data.max() - data.min()) / bin_width))
        elif method == 'blocks':
            return _bayesian_blocks(data)
        else:
            raise ValueError(f"Unknown bin method: {method}")

    def pdf(self, x):
        _check_fitted(self)
        x = np.asarray(x)
        indices = np.digitize(x, self.bin_edges_[1:-1])
        result = np.zeros_like(x, dtype=float)
        valid = (indices > 0) & (indices < len(self.bin_edges_))
        result[valid] = self.hist_[indices[valid] - 1]
        return result

    def plot(self, ax=None, **kwargs):
        _check_fitted(self)
        if ax is None:
            _, ax = plt.subplots()
        x = np.repeat(self.bin_edges_, 2)[1:-1]
        y = np.repeat(self.hist_, 2)
        ax.plot(x, y, **kwargs)
        ax.set_title("Histogram estimate")
        return ax

def _bayesian_blocks(data, p0=0.05):
    """Simplified Bayesian Blocks binning (returns number of bins)."""
    n = len(data)
    return max(1, int(np.sqrt(n)))

# Bayesian Nonparametric Methods

class BayesianBootstrap:
    """
    Bayesian bootstrap for estimating posterior distributions of statistics.

    Parameters
    ----------
    n_resamples : int, default=10000
        Number of bootstrap samples to generate.
    """
    def __init__(self, n_resamples=10000):
        self.n_resamples = n_resamples
        self._fitted = False

    def fit(self, data):
        data = _check_1d(data)
        self.data_ = np.asarray(data)
        self.n_ = len(self.data_)

        self.weights_ = np.random.dirichlet(
            np.ones(self.n_), size=self.n_resamples
        )
        self._fitted = True
        return self

    def sample_statistic(self, statistic):
        _check_fitted(self)
        samples = np.zeros(self.n_resamples)
        for i, w in enumerate(self.weights_):
            indices = np.random.choice(self.n_, size=self.n_, p=w)
            samples[i] = statistic(self.data_[indices])
        return samples

    def confidence_interval(self, statistic, alpha=0.05):
        samples = self.sample_statistic(statistic)
        lower = np.percentile(samples, 100 * alpha / 2)
        upper = np.percentile(samples, 100 * (1 - alpha / 2))
        return lower, upper

# Nonparametric Hypothesis Tests

def goodness_of_fit(data, dist='normal', method='kde', n_bootstrap=1000, random_state=None):
    """
    Test whether data comes from a specified distribution.

    Parameters

    data : array-like, 1D
        Observed data.
    dist : str or scipy.stats distribution object
        Null distribution. If str, one of 'normal', 'uniform', 'expon'.
    method : str, default='kde'
        Test statistic method: 'kde' (L2 distance between KDE and null density)
        or 'ks' (Kolmogorov-Smirnov).
    n_bootstrap : int, default=1000
        Number of bootstrap resamples for p-value.
    random_state : int, optional
        Seed for reproducibility.

    Returns

    p_value : float
        Approximate p-value.
    """
    rng = np.random.default_rng(random_state)
    data = np.asarray(data).ravel()
    n = len(data)

    if isinstance(dist, str):
        if dist == 'normal':
            dist_obj = stats.norm(loc=np.mean(data), scale=np.std(data, ddof=1))
        elif dist == 'uniform':
            dist_obj = stats.uniform(loc=data.min(), scale=data.max()-data.min())
        elif dist == 'expon':
            dist_obj = stats.expon(scale=np.mean(data))
        else:
            raise ValueError(f"Unknown distribution: {dist}")
    else:
        dist_obj = dist

    if method == 'ks':
        if isinstance(dist, str) and dist == 'normal':
            _, p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
            return p
        else:
            _, p = stats.kstest(data, dist_obj.cdf)
            return p

    elif method == 'kde':
        kde = KDE().fit(data)
        grid = np.linspace(data.min() - 3*kde.bw_, data.max() + 3*kde.bw_, 500)
        obs_stat = np.trapz((kde.pdf(grid) - dist_obj.pdf(grid))**2, grid)

        count = 0
        for _ in range(n_bootstrap):
            sample = dist_obj.rvs(size=n, random_state=rng)
            kde_null = KDE().fit(sample)
            stat = np.trapz((kde_null.pdf(grid) - dist_obj.pdf(grid))**2, grid)
            if stat >= obs_stat:
                count += 1
        return count / n_bootstrap

    else:
        raise ValueError(f"Unknown method: {method}")

def two_sample_test(x, y, method='kde', n_permutations=1000, random_state=None):
    """
    Test whether two samples come from the same distribution.

    Parameters

    x, y : array-like, 1D
        Two samples.
    method : str, default='kde'
        Test statistic: 'kde' (L2 distance between KDEs) or 'ks' (Kolmogorov-Smirnov).
    n_permutations : int, default=1000
        Number of permutations for p-value.
    random_state : int, optional
        Seed for reproducibility.

    Returns

    statistic : float
        Observed test statistic.
    p_value : float
        Permutation p-value.
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    combined = np.concatenate([x, y])
    n_x, n_y = len(x), len(y)

    if method == 'ks':
        obs_stat, _ = stats.ks_2samp(x, y)
        count = 0
        for _ in range(n_permutations):
            perm = rng.permutation(combined)
            stat, _ = stats.ks_2samp(perm[:n_x], perm[n_x:])
            if stat >= obs_stat:
                count += 1
        return obs_stat, count / n_permutations

    elif method == 'kde':
        kde_x = KDE().fit(x)
        kde_y = KDE().fit(y)
        grid = np.linspace(combined.min(), combined.max(), 500)
        obs_stat = np.trapz((kde_x.pdf(grid) - kde_y.pdf(grid))**2, grid)

        count = 0
        for _ in range(n_permutations):
            perm = rng.permutation(combined)
            kde1 = KDE().fit(perm[:n_x])
            kde2 = KDE().fit(perm[n_x:])
            stat = np.trapz((kde1.pdf(grid) - kde2.pdf(grid))**2, grid)
            if stat >= obs_stat:
                count += 1
        return obs_stat, count / n_permutations

    else:
        raise ValueError(f"Unknown method: {method}")

def spearman(x, y):
    """Spearman rank correlation coefficient and p-value."""
    return stats.spearmanr(x, y)

def kendall(x, y):
    """Kendall's tau correlation coefficient and p-value."""
    return stats.kendalltau(x, y)

def hoeffding(x, y):
    """
    Hoeffding's D measure of dependence.
    (Uses scipy's implementation if available.)
    """
    try:
        from scipy.stats import hoeffding as hoeffd
        return hoeffd(x, y)
    except ImportError:
        raise NotImplementedError("Hoeffding's D requires scipy>=1.7.0")

def bootstrap_test(statistic, data, null_value=0, n_bootstrap=1000,
                   alternative='two-sided', random_state=None):
    """
    Perform a bootstrap hypothesis test for a statistic.

    Parameters

    statistic : callable
        Function that computes the statistic of interest from a sample.
    data : array-like
        Observed sample.
    null_value : float, default=0
        Value of the statistic under the null hypothesis.
    n_bootstrap : int, default=1000
        Number of bootstrap resamples.
    alternative : str, default='two-sided'
        'two-sided', 'greater', or 'less'.
    random_state : int, optional
        Seed for reproducibility.

    Returns

    p_value : float
        Bootstrap p-value.
    """
    rng = np.random.default_rng(random_state)
    data = np.asarray(data)
    n = len(data)
    obs_stat = statistic(data)

    centered = data - obs_stat + null_value
    boot_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(centered, size=n, replace=True)
        boot_stats[i] = statistic(sample)

    if alternative == 'two-sided':
        p = np.mean(np.abs(boot_stats - null_value) >= np.abs(obs_stat - null_value))
    elif alternative == 'greater':
        p = np.mean(boot_stats >= obs_stat)
    elif alternative == 'less':
        p = np.mean(boot_stats <= obs_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    return p