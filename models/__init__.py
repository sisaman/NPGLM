import numpy as np
from scipy.optimize import minimize
from scipy.stats import expon, rayleigh, gompertz, pareto
from sklearn.base import BaseEstimator


class Model(BaseEstimator):
    def __init__(self):
        self.w = None  # vector of coefficients
        self.f = None  # negative log likelihood for training data

    def fit(self, X, Y, T): pass  # fit the data to the model

    def mean(self, X): pass

    def quantile(self, X, q): pass  # returns the answer to the quantile query


def augment(X: np.ndarray):
    n = X.shape[0]
    v = np.ones((n, 1))
    return np.append(v, X, axis=1)


def optimize(nloglf, p0):
    def hes(w): return nloglf(w)[2]

    result = minimize(nloglf, p0, method='trust-ncg', jac=True, hess=hes)
    return result.x, result.fun


def generate_data(w, n, dist_rnd):
    d = len(w) - 1
    X = np.random.randn(n, d)
    T = dist_rnd(np.exp(-augment(X).dot(w)))
    index = np.argsort(T, axis=0).ravel()
    return X[index, :], T[index]


def gom_rnd(a):
    u = np.random.rand(a.shape[0], a.shape[1])
    return np.log(1 - np.log(1 - u) / a)


def get_dist_rnd(dist):
    return {
        'exp': np.random.exponential,
        'ray': np.random.rayleigh,
        'pow': np.random.pareto,
        'gom': lambda a: gom_rnd(1 / a)
    }[dist]


def get_dist_pdf(dist):
    return {
        'exp': expon.pdf,
        'ray': rayleigh.pdf,
        'pow': pareto.pdf,
        'gom': lambda x, scale: gompertz.pdf(x, 1 / scale)
    }[dist]
