def _log_multivariate_density(X, means, covars):
    """
      Class conditional density:
        P(x | mu, Sigma) = 1/((2pi)^d/2 * |Sigma|^1/2) * exp(-1/2 * (x-mu)^T * Sigma^-1 * (x-mu))

      log of class conditional density:
        log P(x | mu, Sigma) = -1/2*(d*log(2pi) + log(|Sigma|) + (x-mu)^T * Sigma^-1 * (x-mu))
    """
    n_samples, n_dim = X.shape
    n_components = means.shape[0]

    assert(means.shape[0] == covars.shape[0])

    log_proba = np.empty(shape=(n_samples, n_components), dtype=float)
    for i, (mu, cov) in enumerate(zip(means, covars)):
        try:
            cov_chol = scipy.linalg.cholesky(cov, lower=True)
        except scipy.linalg.LinAlgError:
            try:
                cov_chol = scipy.linalg.cholesky(cov + Lambda*np.eye(n_dim), lower=True)
            except:
                raise ValueError("Triangular Matrix Decomposition not performed!\n")

        cov_log_det = 2 * np.sum(np.log(np.diagonal(cov_chol)))

        try:
            cov_solve = scipy.linalg.solve_triangular(cov_chol, (X - mu).T, lower=True).T
        except:
            raise ValueError("Solve_triangular not perormed!\n")

        log_proba[:, i] = -0.5 * (np.sum(cov_solve ** 2, axis=1) + \
                         n_dim * np.log(2 * np.pi) + cov_log_det)

    return(log_proba)