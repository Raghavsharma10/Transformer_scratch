def spline_base3d( width, height, depth, nr_knots_x = 10.0, nr_knots_y = 10.0,
        nr_knots_z=10, spline_order = 3, marginal_x = None, marginal_y = None, 
        marginal_z = None):
    """Computes a set of 3D spline basis functions. 
    
    For a description of the parameters see spline_base2d.
    """  
    if not nr_knots_z < depth:
        raise RuntimeError("Too many knots for size of the base")
    basis2d, (knots_x, knots_y) = spline_base2d(height, width, nr_knots_x, 
            nr_knots_y, spline_order, marginal_x, marginal_y)
    if marginal_z is not None:
        knots_z = knots_from_marginal(marginal_z, nr_knots_z, spline_order)
    else:
        knots_z = augknt(np.linspace(0,depth+1, nr_knots_z), spline_order)
    z_eval = np.arange(1,depth+1).astype(float)
    spline_setz = spcol(z_eval, knots_z, spline_order)
    bspline = np.zeros((basis2d.shape[0]*len(z_eval), height*width*depth))
    basis_nr = 0
    for spline_a in spline_setz.T:
        for spline_b in basis2d:
            spline_b = spline_b.reshape((height, width))
            bspline[basis_nr, :] = (spline_b[:,:,np.newaxis] * spline_a[:]).flat
            basis_nr +=1
    return bspline, (knots_x, knots_y, knots_z)