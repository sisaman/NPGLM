import logging
import numpy as np
from sklearn.model_selection import train_test_split

from models import Model
from models import optimize


class NpGlm(Model):
    def __init__(self):
        super().__init__()
        self.t = None
        self.H = None

    def fit(self, X, Y, T, max_iter=2000):

        X, X_val, Y, Y_val, T, T_val = train_test_split(X, Y, T, test_size=0.1, stratify=Y)
        
        idx = np.argsort(T)
        X = X[idx]
        Y = Y[idx]
        T = T[idx]

        self.t = T
        d = X.shape[1]
        self.w = np.zeros((d, 1))
        f_old = np.inf
        loss_fn = lambda w: self.loss(w, X, Y, self.H)

        for i in range(max_iter):
            self.H = self.cumulative_h(X, Y)
            self.w, self.f = optimize(loss_fn, self.w)

            print(f'Iteration {i}: \t train loss: {self.f / len(T):.4f} \t val error: {self.evaluate(X_val, Y_val, T_val):.4f}')

            if abs(self.f - f_old) < 1e-3:
                break

            f_old = self.f

    def cumulative_h(self, X, Y):
        E = np.exp(X.dot(self.w))
        cumexp = np.cumsum(E[::-1])[::-1]
        frac = Y / cumexp
        return np.cumsum(frac)

    def mean(self, X):
        E = np.exp(X.dot(self.w))
        HE = E[None].T * self.H
        EHE = np.exp(-HE)
        T = np.append(0, self.t)
        EHET = EHE * np.diff(T)
        return EHET.sum(axis=1)

    def quantile(self, X, q):
        Hta = -np.log(1 - q) / np.exp(X.dot(self.w))
        n = len(Hta)
        m = len(self.H)
        T = np.zeros((n, 1))
        for i in range(n):
            k = np.searchsorted(self.H, Hta[i], side='right')
            k = k - (2 if k == m else 1)
            delta_t = self.t[k + 1] - self.t[k]
            if delta_t == 0:
                T[i] = self.t[k]
            else:
                T[i] = (self.t[k + 1] - self.t[k]) * (Hta[i] - self.H[k]) / (self.H[k + 1] - self.H[k]) + self.t[k]

        return T

    def evaluate(self, X, Y, T):
        T_pred = self.quantile(X, q=0.5)
        MAE = np.mean(np.abs(T_pred[Y] - T[Y]))
        return MAE

    @staticmethod
    def loss(w, X, Y, H):
        """
        negative log likelihood (complete)
        refer to formulations of NP-GLM
        """
        Xw = np.dot(X, w)
        E = np.exp(Xw)
        HE = H * E
        p = X * (HE - Y)[:, None]
        f = np.sum(HE - Y * Xw, axis=0)
        g = np.sum(p, axis=0)
        h = np.dot(X.T, (X * HE[:, None]))
        return f, g, h


def main():
    model = NpGlm()
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    Y = np.array([True, True, True, False])
    T = np.array([1, 2, 3, 4])

    model.fit(X, Y, T)
    print(model.f)
    print(model.quantile(X, 0.5))


if __name__ == '__main__':
    main()
