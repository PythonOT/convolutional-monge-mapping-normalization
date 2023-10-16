import numpy as np

import scipy.signal
import scipy.fft as sp_fft
from abc import ABC


class CMMN(ABC):
    """Base class for CMMN."""
    def __init__(self, filter_size=128, fs=100, weights=None):
        super().__init__()
        self.filter_size = filter_size
        self.fs = fs
        self.weights = weights
        self.barycenter = None

    def fit(self, X):
        """Fit the barycenter.

        Parameters
        ----------
        X : list, shape=(K, C, T) or (K, N, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        """
        if len(X[0].shape) == 3:
            # Reduce the number of dimension to (C, T)
            X = [np.concatenate(X[i], axis=-1) for i in range(len(X))]

        self.compute_barycenter(X)

        return self

    def transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : list, shape=(K, C, T) or (K, N, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        """
        reduce = False
        if len(X[0].shape) == 3:
            window_size = X[0].shape[-1]
            # Reduce the number of dimension to (C, T)
            X = [np.concatenate(X[i], axis=-1) for i in range(len(X))]
            reduce = True
        H = self.compute_filter(X)
        X = self.compute_convolution(X, H)

        if reduce:
            X = [self._epoching(X[i], window_size) for i in range(len(X))]
        return X

    def fit_transform(self, X):
        """Fit the model and transform the data.

        Parameters
        ----------
        X : list, shape=(K, C, T) or (K, N, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        """
        self.fit(X)
        return self.transform(X)

    def compute_barycenter(self, X):
        """Filter the signal with given filter.

        Parameters
        ----------
        X : list, shape=(K, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        """
        self.barycenter = self._temporal_barycenter(X)

    def _temporal_barycenter(self, X):
        K = len(X)
        psd = [
            scipy.signal.welch(X[i], nperseg=self.filter_size, fs=self.fs)[1]
            for i in range(K)
        ]
        psd = np.array(psd)
        if self.weights is None:
            weights = np.ones(psd.shape, dtype=X[0].dtype) / K

        barycenter = np.sum(weights * np.sqrt(psd), axis=0) ** 2
        return barycenter

    def compute_filter(self, X):
        """Compute filter to mapped the source data to the barycenter.

        This function compute the filter to mapped the source data to target
        frequency barycenter. One target need to be given to compute the
        filter.

        X : list, shape=(K, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        """
        H = self._compute_temporal_filter(X)
        return np.fft.fftshift(H, axes=-1)

    def _compute_temporal_filter(self, X):
        K = len(X)
        psd = [
            scipy.signal.welch(X[i], nperseg=self.filter_size, fs=self.fs)[1]
            for i in range(K)
        ]

        if self.barycenter is None:
            raise ValueError("Barycenter need to be computed first")
        D = np.sqrt(self.barycenter) / np.sqrt(psd)
        H = sp_fft.irfftn(D, axes=-1)

        return H

    def compute_convolution(self, X, H):
        """Filter the signal with given filter.

        Parameters
        ----------
        X : list, shape=(K, C, T)
            List of data signals for each domain. Each domain
            does not have the same time length.
        H : array, shape=(K, C, filter_size)
            Filters.
        """
        return [self._temporal_convolution(X[i], H[i]) for i in range(len(H))]

    def _temporal_convolution(self, X, H):
        X_norm = [
            np.convolve(X[chan], H[chan], mode="same")
            for chan in range(len(H))
        ]
        X_norm = np.array(X_norm)

        return X_norm

    def _epoching(self, X, size):
        """Create a epoch of size `size` on the data `X`.

        Parameters
        ----------
        X : array, shape=(C, T)
            Data.
        size : int
            Size of the window.

        Returns
        -------
        array, shape=(n_epochs, C, size)
        """
        data = []
        start = 0
        end = size
        step = size
        length = X.shape[-1]
        while end <= length:
            data.append(X[:, start:end])
            start += step
            end += step
        return np.array(data)
