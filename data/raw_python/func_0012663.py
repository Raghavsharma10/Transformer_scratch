def spline_base2d(width, height, nr_knots_x = 20.0, nr_knots_y = 20.0, 
        spline_order = 5, marginal_x = None, marginal_y = None):
    """Computes a set of 2D spline basis functions. 
    
    The basis functions cover the entire space in height*width and can 
    for example be used to create fixation density maps. 

    Input:
        width: int
            width  of each basis
        height: int 
            height of each basis
        nr_knots_x: int
            of knots in x (width) direction.
        nr_knots_y: int
            of knots in y (height) direction.
        spline_order: int
            Order of the spline.
        marginal_x: array, optional
            Estimate of marginal distribution of the input to be fitted
            along the x-direction (width). If given, it is used to determine 
            the positioning of knots, each knot will cover the same amount 
            of probability mass. If not given, knots are equally spaced.
        marginal_y: array, optional
            Marginal distribution along the y-direction (height). If
            given, it is used to determine the positioning of knots.
            Each knot will cover the same amount of probability mass.
    Output:
        basis: Matrix 
            Matrix of size n*(width*height) that contains in each row
            one vectorized basis. 
        knots: Tuple 
            (x,y) are knot arrays that show the placement of knots.
    """
    if not (nr_knots_x<width and nr_knots_y<height):
        raise RuntimeError("Too many knots for size of the base")
    if marginal_x is None:
        knots_x         = augknt(np.linspace(0,width+1,nr_knots_x), spline_order)
    else:
        knots_x = knots_from_marginal(marginal_x, nr_knots_x, spline_order) 
    if marginal_y is None:
        knots_y         = augknt(np.linspace(0,height+1, nr_knots_y), spline_order)
    else:
        knots_y = knots_from_marginal(marginal_y, nr_knots_y, spline_order)
    x_eval = np.arange(1,width+1).astype(float)
    y_eval = np.arange(1,height+1).astype(float)    
    spline_setx = spcol(x_eval, knots_x, spline_order)
    spline_sety = spcol(y_eval, knots_y, spline_order)
    nr_coeff = [spline_sety.shape[1], spline_setx.shape[1]]
    dim_bspline = [nr_coeff[0]*nr_coeff[1], len(x_eval)*len(y_eval)]
    # construct 2D B-splines 
    nr_basis = 0
    bspline = np.zeros(dim_bspline)
    for IDX1 in range(0,nr_coeff[0]):
        for IDX2 in range(0, nr_coeff[1]):
            rand_coeff  = np.zeros((nr_coeff[0] , nr_coeff[1]))
            rand_coeff[IDX1,IDX2] = 1
            tmp = np.dot(spline_sety,rand_coeff)
            bspline[nr_basis,:] = np.dot(tmp,spline_setx.T).reshape((1,-1))
            nr_basis = nr_basis+1
    return bspline, (knots_x, knots_y)