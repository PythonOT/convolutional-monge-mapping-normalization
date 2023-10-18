import numpy as np
import scipy.fft as sp_fft

from cmmn import CMMN

import pytest
from numpy.testing import assert_almost_equal


@pytest.mark.parametrize(
    "filter_size, n_channels", [
        (4, 3),
        (8, 5),
        (4, 3),
        (8, 5),
    ]
)
def test_cmmn(filter_size, n_channels):
    rng = np.random.RandomState(42)
    n_domains = 3
    n_samples = 10
    n_times = 20
    X = rng.rand(n_domains, n_samples, n_channels, n_times)

    cmmn = CMMN(filter_size=filter_size)
    cmmn.fit(X)

    assert sp_fft.irfftn(cmmn.barycenter).shape == (n_channels, filter_size)

    X_transform = cmmn.transform(X)
    cmmn = CMMN(filter_size=filter_size)
    X_transform_2 = cmmn.fit_transform(X)

    assert_almost_equal(X_transform, X_transform_2)

    cmmn = CMMN(filter_size=filter_size)
    cmmn.fit(X)
    cmmn_2 = CMMN(filter_size=filter_size)
    X_ = [np.concatenate(X[i], axis=-1) for i in range(len(X))]
    cmmn_2.fit(X_)

    assert_almost_equal(cmmn.barycenter, cmmn_2.barycenter)
