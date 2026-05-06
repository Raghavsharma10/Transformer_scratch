def deriv(f,c,dx=0.0001):
    """
    deriv(f,c,dx)  --> float
    
    Returns f'(x), computed as a symmetric difference quotient.
    """
    return (f(c+dx)-f(c-dx))/(2*dx)