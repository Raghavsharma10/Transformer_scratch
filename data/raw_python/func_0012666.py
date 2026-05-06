def spcol(x,knots,spline_order):
    """Computes the spline colocation matrix for knots in x.
    
    The spline collocation matrix contains all m-p-1 bases 
    defined by knots. Specifically it contains the ith basis
    in the ith column.
    
    Input:
        x: vector to evaluate the bases on
        knots: vector of knots 
        spline_order: order of the spline
    Output:
        colmat: m x m-p matrix
            The colocation matrix has size m x m-p where m 
            denotes the number of points the basis is evaluated
            on and p is the spline order. The colums contain 
            the ith basis of knots evaluated on x.
    """
    colmat = np.nan*np.ones((len(x),len(knots) - spline_order-1))
    for i in range(0,len(knots) - spline_order -1):
        colmat[:,i] = spline(x,knots,spline_order,i)
    return colmat