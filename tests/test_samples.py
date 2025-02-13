from simple_gaussian.likelihood import ln_analytic_evidence
import numpy as np


def test_ln_analytic_evidence():
    ndim = 2
    inv_cov = np.zeros((ndim, ndim))
    diag_cov = np.ones(ndim)
    np.fill_diagonal(inv_cov, diag_cov)

    lnz = ln_analytic_evidence(ndim, inv_cov)
    np.testing.assert_allclose(
        lnz,
        1.8378770664093453,
        rtol=1e-3,
        err_msg="Analytic evidence is not correct",
    )
