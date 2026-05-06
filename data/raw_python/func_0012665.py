def spline(x,knots,p,i=0.0):
    """Evaluates the ith spline basis given by knots on points in x"""
    assert(p+1<len(knots))
    return np.array([N(float(u),float(i),float(p),knots) for u in x])