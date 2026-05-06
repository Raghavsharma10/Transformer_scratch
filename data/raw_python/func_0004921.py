def errtrapz(x, yerr):
    """Error of the trapezoid formula
    Inputs:
        x: the abscissa
        yerr: the error of the dependent variable

    Outputs:
        the error of the integral
    """
    x = np.array(x)
    assert isinstance(x, np.ndarray)
    yerr = np.array(yerr)
    return 0.5 * np.sqrt((x[1] - x[0]) ** 2 * yerr[0] ** 2 +
                         np.sum((x[2:] - x[:-2]) ** 2 * yerr[1:-1] ** 2) +
                         (x[-1] - x[-2]) ** 2 * yerr[-1] ** 2)