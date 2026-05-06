def spline_base1d(length, nr_knots = 20, spline_order = 5, marginal = None):
    """Computes a 1D spline basis
    
    Input:
        length: int
            length  of each basis
        nr_knots: int
            Number of knots, i.e. number of basis functions.
        spline_order: int
            Order of the splines.
        marginal: array, optional
            Estimate of the marginal distribution of the input to be fitted. 
            If given, it is used to determine the positioning of knots, each 
            knot will cover the same amount of probability mass. If not given,
            knots are equally spaced.
    """
    if marginal is None:
        knots = augknt(np.linspace(0,length+1, nr_knots), spline_order)
    else:
        knots = knots_from_marginal(marginal, nr_knots, spline_order)
        
    x_eval = np.arange(1,length+1).astype(float)
    Bsplines    = spcol(x_eval,knots,spline_order)
    return Bsplines, knots