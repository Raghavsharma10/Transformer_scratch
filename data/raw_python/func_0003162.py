def get_points_and_weights(w_func=lambda x : np.ones(x.shape), 
    left=-1.0, right=1.0, num_points=5, n=4096):
    """Quadratude points and weights for a weighting function.

    Points and weights for approximating the integral 
        I = \int_left^right f(x) w(x) dx
    given the weighting function w(x) using the approximation
        I ~ w_i f(x_i)

    Args:
        w_func: The weighting function w(x). Must be a function that takes
            one argument and is valid over the open interval (left, right).
        left: The left boundary of the interval
        right: The left boundary of the interval
        num_points: number of integration points to return
        n: the number of points to evaluate w_func at.

    Returns:
        A tuple (points, weights) where points is a sorted array of the
        points x_i and weights gives the corresponding weights w_i.
    """
    
    dx = (float(right)-left)/n
    z = np.hstack(np.linspace(left+0.5*dx, right-0.5*dx, n))
    w = dx*w_func(z)  
   
    (a, b) = discrete_gautschi(z, w, num_points)
    alpha = a
    beta = np.sqrt(b)
    
    J = np.diag(alpha)
    J += np.diag(beta, k=-1)
    J += np.diag(beta, k=1)    
    
    (points,v) = np.linalg.eigh(J)
    ind = points.argsort()
    points = points[ind]
    weights = v[0,:]**2 * w.sum()
    weights = weights[ind]
    
    return (points, weights)