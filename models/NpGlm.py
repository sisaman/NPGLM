import numpy as np

from models import Model
from models import optimize


class NpGlm(Model):
    def __init__(self):
        super().__init__()
        self.t = None
        self.H = None

    def fit(self, X, Y, T):  # X,Y, and T must be sorted by T beforehand
        self.t = T
        max_iter = 2000
        # print(max_iter)
        d = X.shape[1]
        self.w = np.zeros((d, 1))
        f_old = np.inf

        for i in range(max_iter):
            self.cumulative_h(X, Y)

            # h = self.h_estimator()

            def nloglf(w):
                return NpGlm.nlogl(w, None, self.H, X, Y, T)

            self.w, self.f = optimize(nloglf, self.w)

            # logging.info('%d\t%f' % (i, self.f / len(T)))
            # if self.conv is not None:
            #    self.conv.append((i, -self.f / len(T)))

            if abs(self.f - f_old) < 1e-3:
                break

            f_old = self.f

    def cumulative_h(self, X, Y):
        E = np.exp(X.dot(self.w))
        cumexp = np.cumsum(E[::-1])[::-1]
        frac = Y / cumexp
        self.H = np.cumsum(frac)

    def h_estimator(self):
        H = np.append(0, self.H)
        T = np.append(0, self.t)
        h = np.diff(H) / np.diff(T)
        h[np.isnan(h)] = 0
        return h

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

    def get_error(self, w, dist):
        w = w[1:].T
        if dist == 'ray':
            w = 2 * w

        res = np.abs(w - self.w)
        return np.mean(res)

    def log_likelihood(self, X, Y, T):
        H = np.interp(T.ravel(), self.t.ravel(), self.H)
        Ha = np.insert(H, 0, 0)
        Ta = np.insert(T, 0, 0)
        h = np.diff(Ha) / np.diff(Ta)
        return -NpGlm.nlogl(self.w, h, H, X, Y, T)[0] / len(T)

    def pdf(self, X, T):
        h = self.h_estimator()
        N = len(T)
        f = np.zeros((N, 1))
        for i in range(N):
            x = X[i,]
            t = T[i]
            k = np.searchsorted(self.t.ravel(), t, side='left')
            wx = self.w.dot(x)
            f[i] = np.exp(wx) * h[k] * np.exp(-self.H[k] * np.exp(wx))
        return f

    @staticmethod
    def nlogl(w, h, H, X, Y, T):
        """
        negative log likelihood (complete)
        refer to formulations of NP-GLM
        """
        # h[h == 0] = 1e-20
        Xw = np.dot(X, w)
        E = np.exp(Xw)
        HE = H * E
        p = X * (HE - Y)[:, None]
        # f = np.sum(HE - Y * (Xw + np.log(h)), axis=0)
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
