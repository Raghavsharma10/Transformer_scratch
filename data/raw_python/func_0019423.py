def _maximization_step(X, posteriors):
    """ 
      Update class parameters as below:
        priors: P(w_i) = sum_x P(w_i | x) ==> Then normalize to get in [0,1]
        Class means: center_w_i = sum_x P(w_i|x)*x / sum_i sum_x P(w_i|x)
    """

    ### Prior probabilities or class weights
    sum_post_proba = np.sum(posteriors, axis=0)
    prior_proba = sum_post_proba / (sum_post_proba.sum() + Epsilon)
    
    ### means
    means = np.dot(posteriors.T, X) / (sum_post_proba[:, np.newaxis] + Epsilon)

    ### covariance matrices
    n_components = posteriors.shape[1]
    n_features = X.shape[1]
    covars = np.empty(shape=(n_components, n_features, n_features), dtype=float)
  
    for i in range(n_components):
        post_i = posteriors[:, i]
        mean_i = means[i]
        diff_i = X - mean_i

        with np.errstate(under='ignore'):
            covar_i = np.dot(post_i * diff_i.T, diff_i) / (post_i.sum() + Epsilon)
        covars[i] = covar_i + Lambda * np.eye(n_features)


    _validate_params(prior_proba, means, covars)
    return(prior_proba, means, covars)