def NegLnLike(x, time, flux, errors, kernel):
    '''
    Returns the negative log-likelihood function and its gradient.

    '''

    gp = GP(kernel, x, white=True)
    gp.compute(time, errors)
    if OLDGEORGE:
        nll = -gp.lnlikelihood(flux)
        # NOTE: There was a bug on this next line! Used to be
        #
        #    ngr = -gp.grad_lnlikelihood(flux) / gp.kernel.pars
        #
        # But I think we want
        #
        # dlogL/dx =     dlogL/dlogx^2       * dlogx^2/dx^2 * dx^2/dx
        #          = gp.grad_lnlikelihood()  *     1/x^2    *   2x
        #          = 2 * gp.grad_lnlikelihood() / x
        #          = 2 * gp.grad_lnlikelihood() / np.sqrt(x^2)
        #          = 2 * gp.grad_lnlikelihood() / np.sqrt(gp.kernel.pars)
        #
        # (with a negative sign out front for the negative gradient).
        # So we probably weren't optimizing the GP correctly! This affects
        # all campaigns through C13. It's not a *huge* deal, since the sign
        # of the gradient was correct and the model isn't that sensitive to
        # the value of the hyperparameters, but it may have contributed to
        # the poor performance on super variable stars. In most cases it means
        # the solver takes longer to converge and isn't as good at finding
        # the minimum.
        ngr = -2 * gp.grad_lnlikelihood(flux) / np.sqrt(gp.kernel.pars)
    else:
        nll = -gp.log_likelihood(flux)
        ngr = -2 * gp.grad_log_likelihood(flux) / \
            np.sqrt(np.exp(gp.get_parameter_vector()))

    return nll, ngr