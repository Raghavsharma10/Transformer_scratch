def _PrPz(r0, z0, r1, z1, r2, z2, r3, z3):
    """
    Intersection point for infinite lines.
    
    Parameters
    ----------
    r0 : float
    z0 : float
    r1 : float
    z1 : float
    r2 : float
    z2 : float
    r3 : float
    z3 : float

    Returns
    ----------
    Pr : float    
    Pz : float
    hit : bool

    """
    Pr = ((r0*z1 - z0*r1)*(r2 - r3) - (r0 - r1)*(r2*z3 - r3*z2)) / \
                        ((r0 - r1)*(z2 - z3) - (z0 - z1)*(r2-r3))
    Pz = ((r0*z1 - z0*r1)*(z2 - z3) - (z0 - z1)*(r2*z3 - r3*z2)) / \
                        ((r0 - r1)*(z2 - z3) - (z0 - z1)*(r2-r3))
    
    if Pr >= r0 and Pr <= r1 and Pz >= z0 and Pz <= z1:
        hit = True
    elif Pr <= r0 and Pr >= r1 and Pz >= z0 and Pz <= z1:
        hit = True
    elif Pr >= r0 and Pr <= r1 and Pz <= z0 and Pz >= z1:
        hit = True
    elif Pr <= r0 and Pr >= r1 and Pz <= z0 and Pz >= z1:
        hit = True
    else:
        hit = False
        
    return [Pr, Pz, hit]