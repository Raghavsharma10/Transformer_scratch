def _gauss_funct(p, fjac=None, x=None, y=None, err=None,
                 weights=None):

    """
    Defines the gaussian function to be used as the model.

    """
    if p[2] != 0.0:
        Z = (x - p[1]) / p[2]
        model = p[0] * np.e ** (-Z ** 2 / 2.0)
    else:
        model = np.zeros(np.size(x))

    status = 0
    if weights is not None:
        if err is not None:
            print("Warning: Ignoring errors and using weights.\n")

        return [status, (y - model) * weights]

    elif err is not None:
        return [status, (y - model) / err]

    else:
        return [status, y - model]