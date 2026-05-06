def toy_linear_1d_classification(seed=default_seed):
    """Simple classification data in one dimension for illustrating models."""
    def sample_class(f):
        p = 1. / (1. + np.exp(-f))
        c = np.random.binomial(1, p)
        c = np.where(c, 1, -1)
        return c

    np.random.seed(seed=seed)
    x1 = np.random.normal(-3, 5, 20)
    x2 = np.random.normal(3, 5, 20)
    X = (np.r_[x1, x2])[:, None]
    return {'X': X, 'Y':  sample_class(2.*X), 'F': 2.*X, 'covariates' : ['X'], 'response': [discrete({'positive': 1, 'negative': -1})],'seed' : seed}