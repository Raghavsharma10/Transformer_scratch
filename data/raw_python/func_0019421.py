def _log_likelihood_per_sample(X, means, covars):
    """
      Theta = (theta_1, theta_2, ... theta_M)
      Likelihood of mixture parameters given data: L(Theta | X) = product_i P(x_i | Theta)
      log likelihood: log L(Theta | X) = sum_i log(P(x_i | Theta))

      and note that p(x_i | Theta) = sum_j prior_j * p(x_i | theta_j)


      Probability of sample x being generated from component i:
         P(w_i | x) = P(x|w_i) * P(w_i) / P(X)
           where P(X) = sum_i P(x|w_i) * P(w_i)

       Here post_proba = P/(w_i | x)
        and log_likelihood = log(P(x|w_i))
    """
   
    logden = _log_multivariate_density(X, means, covars) 

    logden_max = logden.max(axis=1)
    log_likelihood = np.log(np.sum(np.exp(logden.T - logden_max) + Epsilon, axis=0))
    log_likelihood += logden_max

    post_proba = np.exp(logden - log_likelihood[:, np.newaxis])

    return (log_likelihood, post_proba)