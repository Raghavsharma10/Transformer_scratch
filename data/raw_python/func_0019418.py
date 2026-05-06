def _init_mixture_params(X, n_mixtures, init_method):
    """ 
      Initialize mixture density parameters with 
        equal priors
        random means
        identity covariance matrices
    """

    init_priors = np.ones(shape=n_mixtures, dtype=float) / n_mixtures

    if init_method == 'kmeans':
        km = _kmeans.KMeans(n_clusters = n_mixtures, n_trials=20)
        km.fit(X)
        init_means = km.centers_ 
    else:
        inx_rand = np.random.choice(X.shape[0], size=n_mixtures)
        init_means = X[inx_rand,:]
 
  
    if np.any(np.isnan(init_means)):
        raise ValueError("Init means are NaN! ") 

    n_features = X.shape[1]
    init_covars = np.empty(shape=(n_mixtures, n_features, n_features), dtype=float)
    for i in range(n_mixtures):
        init_covars[i] = np.eye(n_features)

    return(init_priors, init_means, init_covars)