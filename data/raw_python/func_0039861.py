def fit_double_lorgauss(bins,h,Ntry=5):
    """Uses lmfit to fit a "Double LorGauss" distribution to a provided histogram.

    Uses a grid of starting guesses to try to avoid local minima.

    Parameters
    ----------
    bins, h : array-like
        Bins and heights of a histogram, as returned by, e.g., `np.histogram`.

    Ntry : int, optional
        Spacing of grid for starting guesses.  Will try `Ntry**2` different
        initial values of the "Gaussian strength" parameters `G1` and `G2`.

    Returns
    -------
    parameters : tuple
        Parameters of best-fit "double LorGauss" distribution.

    Raises
    ------
    ImportError
        If the lmfit module is not available.
    """
    try:
        from lmfit import minimize, Parameters, Parameter, report_fit
    except ImportError:
        raise ImportError('you need lmfit to use this function.')
        
    #make sure histogram is normalized
    h /= np.trapz(h,bins)

    #zero-pad the ends of the distribution to keep fits positive
    N = len(bins)
    dbin = (bins[1:]-bins[:-1]).mean()
    newbins = np.concatenate((np.linspace(bins.min() - N/10*dbin,bins.min(),N/10),
                             bins,
                             np.linspace(bins.max(),bins.max() + N/10*dbin,N/10)))
    newh = np.concatenate((np.zeros(N/10),h,np.zeros(N/10)))

    
    mu0 = bins[np.argmax(newh)]
    sig0 = abs(mu0 - newbins[np.argmin(np.absolute(newh - 0.5*newh.max()))])

    def set_params(G1,G2):
        params = Parameters()
        params.add('mu',value=mu0)
        params.add('sig1',value=sig0)
        params.add('sig2',value=sig0)
        params.add('gam1',value=sig0/10)
        params.add('gam2',value=sig0/10)
        params.add('G1',value=G1)
        params.add('G2',value=G2)
        return params

    sum_devsq_best = np.inf
    outkeep = None
    for G1 in np.linspace(0.1,1.9,Ntry):
        for G2 in np.linspace(0.1,1.9,Ntry):
            params = set_params(G1,G2)
        
            def residual(ps):
                pars = (params['mu'].value,
                        params['sig1'].value,
                        params['sig2'].value,
                        params['gam1'].value,
                        params['gam2'].value,
                        params['G1'].value,
                        params['G2'].value)
                hmodel = double_lorgauss(newbins,pars)
                return newh-hmodel

            out = minimize(residual,params)
            pars = (out.params['mu'].value,out.params['sig1'].value,
                    out.params['sig2'].value,out.params['gam1'].value,
                    out.params['gam2'].value,out.params['G1'].value,
                    out.params['G2'].value)

            sum_devsq = ((newh - double_lorgauss(newbins,pars))**2).sum()
            #print 'devs = %.1f; initial guesses for G1, G2; %.1f, %.1f' % (sum_devsq,G1, G2)
            if sum_devsq < sum_devsq_best:
                sum_devsq_best = sum_devsq
                outkeep = out
            

    return (outkeep.params['mu'].value,abs(outkeep.params['sig1'].value),
            abs(outkeep.params['sig2'].value),abs(outkeep.params['gam1'].value),
            abs(outkeep.params['gam2'].value),abs(outkeep.params['G1'].value),
            abs(outkeep.params['G2'].value))