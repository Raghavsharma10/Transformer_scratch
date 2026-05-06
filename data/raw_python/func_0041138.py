def newton(f,c,tol=0.0001,restrict=None):
    """
    newton(f,c) --> float
    
    Returns the x closest to c such that f(x) = 0
    """
    #print(c)
    if restrict:
        lo,hi = restrict
        if c < lo or c > hi:
            print(c)
            c = random*(hi-lo)+lo

    if fuzzyequals(f(c),0,tol):
        return c
    else:
        try:
            return newton(f,c-f(c)/deriv(f,c,tol),tol,restrict)
        except:
            return None