def _sinusoid(x, p, L, y):
    """ Return the sinusoid cont func evaluated at input x for the continuum.

    Parameters
    ----------
    x: float or np.array
        data, input to function
    p: ndarray
        coefficients of fitting function
    L: float
        width of x data 
    y: float or np.array
        output data corresponding to input x

    Returns
    -------
    func: float
        function evaluated for the input x
    """
    N = int(len(p)/2)
    n = np.linspace(0, N, N+1)
    k = n*np.pi/L
    func = 0
    for n in range(0, N):
        func += p[2*n]*np.sin(k[n]*x)+p[2*n+1]*np.cos(k[n]*x)
    return func