from simple_gaussian.likelihood import ln_analytic_evidence
import numpy as np


def test_ln_analytic_evidence():
    ndim = 2
    inv_cov = np.zeros((ndim, ndim))
    diag_cov = np.ones(ndim)
    np.fill_diagonal(inv_cov, diag_cov)

    lnz = ln_analytic_evidence(ndim, inv_cov)
    assert lnz == -2.837877066409345, "Analytic evidence is not correct"
