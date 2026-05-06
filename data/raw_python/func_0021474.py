def sample_gtf(data, D, k, likelihood='gaussian', prior='laplace',
                           lambda_hyperparams=None, lam_walk_stdev=0.01, lam0=1.,
                           dp_hyperparameter=None, w_hyperparameters=None,
                           iterations=7000, burn=2000, thin=10,
                           robust=False, empirical=False,
                           verbose=False):
    '''Generate samples from the generalized graph trend filtering distribution via a modified Swendsen-Wang slice sampling algorithm.
    Options for likelihood: gaussian, binomial, poisson. Options for prior: laplace, doublepareto.'''
    Dk = get_delta(D, k)
    dk_rows, dk_rowbreaks, dk_cols, dk_vals = decompose_delta(Dk)

    if likelihood == 'gaussian':
        y, w = data
    elif likelihood == 'binomial':
        trials, successes = data
    elif likelihood == 'poisson':
        obs = data
    else:
        raise Exception('Unknown likelihood type: {0}'.format(likelihood))

    if prior == 'laplace':
        if lambda_hyperparams == None:
            lambda_hyperparams = (1., 1.)
    elif prior == 'laplacegamma':
        if lambda_hyperparams == None:
            lambda_hyperparams = (1., 1.)
        if dp_hyperparameter == None:
            dp_hyperparameter = 1.
    elif prior == 'doublepareto' or prior == 'doublepareto2':
        if lambda_hyperparams == None:
            lambda_hyperparams = (1.0, 1.0)
        if dp_hyperparameter == None:
            dp_hyperparameter = 0.1
    elif prior == 'cauchy':
        if lambda_hyperparams == None:
            lambda_hyperparams = (1.0, 1.0)
    else:
        raise Exception('Unknown prior type: {0}.'.format(prior))

    if robust and w_hyperparameters is None:
        w_hyperparameters = (1., 1.)

    # Run the Gibbs sampler
    sample_size = (iterations - burn) / thin
    beta_samples = np.zeros((sample_size, D.shape[1]), dtype='double')
    lam_samples = np.zeros(sample_size, dtype='double')

    if likelihood == 'gaussian':
        if prior == 'laplace':
            gflbayes_gaussian_laplace(len(y), y, w,
                                      dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                      lambda_hyperparams[0], lambda_hyperparams[1],
                                      iterations, burn, thin,
                                      double_matrix_to_c_pointer(beta_samples), lam_samples)
        elif prior == 'laplacegamma':
            if robust:
                gflbayes_gaussian_laplace_gamma_robust(len(y), y, w,
                                          dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                          lambda_hyperparams[0], lambda_hyperparams[1],
                                          dp_hyperparameter,
                                          w_hyperparameters[0], w_hyperparameters[1],
                                          iterations, burn, thin,
                                          double_matrix_to_c_pointer(beta_samples), lam_samples)
            else:    
                gflbayes_gaussian_laplace_gamma(len(y), y, w,
                                          dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                          lambda_hyperparams[0], lambda_hyperparams[1],
                                          dp_hyperparameter,
                                          iterations, burn, thin,
                                          double_matrix_to_c_pointer(beta_samples), lam_samples)
        elif prior == 'doublepareto':
            gflbayes_gaussian_doublepareto(len(y), y, w,
                                      dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                      lambda_hyperparams[0], lambda_hyperparams[1],
                                      lam_walk_stdev, lam0, dp_hyperparameter,
                                      iterations, burn, thin,
                                      double_matrix_to_c_pointer(beta_samples), lam_samples)
        elif prior == 'doublepareto2':
            gflbayes_gaussian_doublepareto2(len(y), y, w,
                                      dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                      lambda_hyperparams[0], lambda_hyperparams[1],
                                      dp_hyperparameter,
                                      iterations, burn, thin,
                                      double_matrix_to_c_pointer(beta_samples), lam_samples)
        elif prior == 'cauchy':
            gflbayes_gaussian_cauchy(len(y), y, w,
                                      dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                      lambda_hyperparams[0], lambda_hyperparams[1],
                                      lam_walk_stdev, lam0,
                                      iterations, burn, thin,
                                      double_matrix_to_c_pointer(beta_samples), lam_samples)
    elif likelihood == 'binomial':
        if prior == 'laplace':
            gflbayes_binomial_laplace(len(trials), trials, successes,
                                      dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                      lambda_hyperparams[0], lambda_hyperparams[1],
                                      iterations, burn, thin,
                                      double_matrix_to_c_pointer(beta_samples), lam_samples)
        elif prior == 'doublepareto':
            gflbayes_binomial_doublepareto(len(trials), trials, successes,
                                      dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                      lambda_hyperparams[0], lambda_hyperparams[1],
                                      lam_walk_stdev, lam0, dp_hyperparameter,
                                      iterations, burn, thin,
                                      double_matrix_to_c_pointer(beta_samples), lam_samples)
        elif prior == 'laplacegamma':
            if empirical:
                gflbayes_empirical_binomial_laplace_gamma(len(trials), trials, successes,
                                          dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                          lam0,
                                          iterations, burn, thin,
                                          double_matrix_to_c_pointer(beta_samples), lam_samples)
            else:
                gflbayes_binomial_laplace_gamma(len(trials), trials, successes,
                                          dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                          lambda_hyperparams[0], lambda_hyperparams[1],
                                          dp_hyperparameter,
                                          iterations, burn, thin,
                                          double_matrix_to_c_pointer(beta_samples), lam_samples)
    elif likelihood == 'poisson':
        if prior == 'laplace':
            gflbayes_poisson_laplace(len(obs), obs,
                                      dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                      lambda_hyperparams[0], lambda_hyperparams[1],
                                      iterations, burn, thin,
                                      double_matrix_to_c_pointer(beta_samples), lam_samples)
        elif prior == 'doublepareto':
            gflbayes_poisson_doublepareto(len(obs), obs,
                                      dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                      lambda_hyperparams[0], lambda_hyperparams[1],
                                      lam_walk_stdev, lam0, dp_hyperparameter,
                                      iterations, burn, thin,
                                      double_matrix_to_c_pointer(beta_samples), lam_samples)
    else:
        raise Exception('Unknown likelihood type: {0}'.format(likelihood))

    return (beta_samples,lam_samples)