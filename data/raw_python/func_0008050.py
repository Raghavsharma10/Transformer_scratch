def _covar_mstep_spherical(*args):
    """Performing the covariance M step for spherical cases"""
    cv = _covar_mstep_diag(*args)
    return np.tile(cv.mean(axis=1)[:, np.newaxis], (1, cv.shape[1]))