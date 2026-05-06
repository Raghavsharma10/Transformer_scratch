def ntwodgaussian_lmfit(params):
    """
    Convert an lmfit.Parameters object into a function which calculates the model.


    Parameters
    ----------
    params : lmfit.Parameters
        Model parameters, can have multiple components.

    Returns
    -------
    model : func
        A function f(x,y) that will compute the model.
    """

    def rfunc(x, y):
        """
        Compute the model given by params, at pixel coordinates x,y

        Parameters
        ----------
        x, y : numpy.ndarray
            The x/y pixel coordinates at which the model is being evaluated

        Returns
        -------
        result : numpy.ndarray
            Model
        """
        result = None
        for i in range(params['components'].value):
            prefix = "c{0}_".format(i)
            # I hope this doesn't kill our run time
            amp = np.nan_to_num(params[prefix + 'amp'].value)
            xo = params[prefix + 'xo'].value
            yo = params[prefix + 'yo'].value
            sx = params[prefix + 'sx'].value
            sy = params[prefix + 'sy'].value
            theta = params[prefix + 'theta'].value
            if result is not None:
                result += elliptical_gaussian(x, y, amp, xo, yo, sx, sy, theta)
            else:
                result = elliptical_gaussian(x, y, amp, xo, yo, sx, sy, theta)
        return result

    return rfunc