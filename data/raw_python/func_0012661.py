def knots_from_marginal(marginal, nr_knots, spline_order):
    """
    Determines knot placement based on a marginal distribution.  

    It places knots such that each knot covers the same amount 
    of probability mass. Two of the knots are reserved for the
    borders which are treated seperatly. For example, a uniform
    distribution with 5 knots will cause the knots to be equally 
    spaced with 25% of the probability mass between each two 
    knots.

    Input:
        marginal: Array
            Estimate of the marginal distribution used to estimate
            knot placement.
        nr_knots: int
            Number of knots to be placed.
        spline_order: int 
            Order of the splines

    Returns:
        knots: Array
            Sequence of knot positions
    """
    cumsum = np.cumsum(marginal)
    cumsum = cumsum/cumsum.max()
    borders = np.linspace(0,1,nr_knots)
    knot_placement = [0] + np.unique([np.where(cumsum>=b)[0][0] for b in borders[1:-1]]).tolist() +[len(marginal)-1]
    knots = augknt(knot_placement, spline_order)
    return knots