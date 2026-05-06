def N(u,i,p,knots):
    """Compute Spline Basis
    
    Evaluates the spline basis of order p defined by knots 
    at knot i and point u.
    """
    if p == 0:
        if knots[i] < u and u <=knots[i+1]:
            return 1.0
        else:
            return 0.0
    else:
        try:
            k = (( float((u-knots[i]))/float((knots[i+p] - knots[i]) )) 
                    * N(u,i,p-1,knots))
        except ZeroDivisionError:
            k = 0.0
        try:
            q = (( float((knots[i+p+1] - u))/float((knots[i+p+1] - knots[i+1])))
                    * N(u,i+1,p-1,knots))
        except ZeroDivisionError:
            q  = 0.0 
        return float(k + q)