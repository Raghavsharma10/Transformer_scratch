def normalize_vector(x, y, z):
    """
    Normalizes vector to produce a unit vector.
    
    Parameters
    ----------
    x : float or array-like
        X component of vector
    y : float or array-like
        Y component of vector
    z : float or array-like
        Z component of vector
        
    Returns
    -------
    x, y, z
        Unit vector x,y,z components
        
    """
    
    mag = np.sqrt(x**2 + y**2 + z**2)
    x = x/mag
    y = y/mag
    z = z/mag
    return x, y, z