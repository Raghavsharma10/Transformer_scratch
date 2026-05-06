def kelly(x1, x2, x1err=[], x2err=[], cerr=[], logify=True,
          miniter=5000, maxiter=1e5, metro=True,
          silent=True):
    """
    Python wrapper for the linear regression MCMC of Kelly (2007).
    Requires pidly (http://astronomy.sussex.ac.uk/~anthonys/pidly/) and
    an IDL license.

    Parameters
    ----------
      x1        : array of floats
                  Independent variable, or observable
      x2        : array of floats
                  Dependent variable
      x1err     : array of floats (optional)
                  Uncertainties on the independent variable
      x2err     : array of floats (optional)
                  Uncertainties on the dependent variable
      cerr      : array of floats (optional)
                  Covariances of the uncertainties in the dependent and
                  independent variables
    """
    import pidly
    
    n = len(x1)
    if len(x2) != n:
        raise ValueError('x1 and x2 must have same length')
    if len(x1err) == 0:
        x1err = numpy.zeros(n)
    if len(x2err) == 0:
        x2err = numpy.zeros(n)
    if logify:
        x1, x2, x1err, x2err = to_log(x1, x2, x1err, x2err)
    idl = pidly.IDL()
    idl('x1 = %s' %list(x1))
    idl('x2 = %s' %list(x2))
    cmd = 'linmix_err, x1, x2, fit'
    if len(x1err) == n:
        idl('x1err = %s' %list(x1err))
        cmd += ', xsig=x1err'
    if len(x2err) == n:
        idl('x2err = %s' %list(x2err))
        cmd += ', ysig=x2err'
    if len(cerr) == n:
        idl('cerr = %s' %list(cerr))
        cmd += ', xycov=cerr'
    cmd += ', miniter=%d, maxiter=%d' %(miniter, maxiter)
    if metro:
        cmd += ', /metro'
    if silent:
        cmd += ', /silent'
    idl(cmd)
    alpha = idl.ev('fit.alpha')
    beta = idl.ev('fit.beta')
    sigma = numpy.sqrt(idl.ev('fit.sigsqr'))
    return alpha, beta, sigma