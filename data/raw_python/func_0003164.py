def ldr(scatterer, h_pol=True):
    """
    Linear depolarizarion ratio (LDR) for the current setup.

    Args:
        scatterer: a Scatterer instance.
        h_pol: If True (default), return LDR_h.
        If False, return LDR_v.

    Returns:
       The LDR.
    """
    Z = scatterer.get_Z()
    if h_pol:
        return (Z[0,0] - Z[0,1] + Z[1,0] - Z[1,1]) / \
               (Z[0,0] - Z[0,1] - Z[1,0] + Z[1,1])
    else:
        return (Z[0,0] + Z[0,1] - Z[1,0] - Z[1,1]) / \
               (Z[0,0] + Z[0,1] + Z[1,0] + Z[1,1])