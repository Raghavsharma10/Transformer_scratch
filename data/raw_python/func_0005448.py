def mle(x1, x2, x1err=[], x2err=[], cerr=[], s_int=True,
        po=(1,0,0.1), verbose=False, logify=True, full_output=False):
    """
    Maximum Likelihood Estimation of best-fit parameters

    Parameters
    ----------
      x1, x2    : float arrays
                  the independent and dependent variables.
      x1err, x2err : float arrays (optional)
                  measurement uncertainties on independent and dependent
                  variables. Any of the two, or both, can be supplied.
      cerr      : float array (same size as x1)
                  covariance on the measurement errors
      s_int     : boolean (default True)
                  whether to include intrinsic scatter in the MLE.
      po        : tuple of floats
                  initial guess for free parameters. If s_int is True, then
                  po must have 3 elements; otherwise it can have two (for the
                  zero point and the slope)
      verbose   : boolean (default False)
                  verbose?
      logify    : boolean (default True)
                  whether to convert the values to log10's. This is to
                  calculate the best-fit power law. Note that the result is
                  given for the equation log(y)=a+b*log(x) -- i.e., the
                  zero point must be converted to 10**a if logify=True
      full_output : boolean (default False)
                  numpy.optimize.fmin's full_output argument

    Returns
    -------
      a         : float
                  Maximum Likelihood Estimate of the zero point. Note that
                  if logify=True, the power-law intercept is 10**a
      b         : float
                  Maximum Likelihood Estimate of the slope
      s         : float (optional, if s_int=True)
                  Maximum Likelihood Estimate of the intrinsic scatter

    """
    from scipy import optimize
    n = len(x1)
    if len(x2) != n:
        raise ValueError('x1 and x2 must have same length')
    if len(x1err) == 0:
        x1err = numpy.ones(n)
    if len(x2err) == 0:
        x2err = numpy.ones(n)
    if logify:
        x1, x2, x1err, x2err = to_log(x1, x2, x1err, x2err)

    f = lambda a, b: a + b * x1
    if s_int:
        w = lambda b, s: numpy.sqrt(b**2 * x1err**2 + x2err**2 + s**2)
        loglike = lambda p: 2 * sum(numpy.log(w(p[1],p[2]))) + \
                            sum(((x2 - f(p[0],p[1])) / w(p[1],p[2])) ** 2) + \
                            numpy.log(n * numpy.sqrt(2*numpy.pi)) / 2
    else:
        w = lambda b: numpy.sqrt(b**2 * x1err**2 + x2err**2)
        loglike = lambda p: sum(numpy.log(w(p[1]))) + \
                            sum(((x2 - f(p[0],p[1])) / w(p[1])) ** 2) / 2 + \
                            numpy.log(n * numpy.sqrt(2*numpy.pi)) / 2
        po = po[:2]
    out = optimize.fmin(loglike, po, disp=verbose, full_output=full_output)
    return out