def orient_averaged_fixed(tm):
    """Compute the T-matrix using variable orientation scatterers.
    
    This method uses a fast Gaussian quadrature and is suitable
    for most use. Uses the set particle orientation PDF, ignoring
    the alpha and beta attributes.

    Args:
        tm: TMatrix (or descendant) instance.

    Returns:
        The amplitude (S) and phase (Z) matrices.
    """
    S = np.zeros((2,2), dtype=complex)
    Z = np.zeros((4,4))
    ap = np.linspace(0, 360, tm.n_alpha+1)[:-1]
    aw = 1.0/tm.n_alpha

    for alpha in ap:
        for (beta, w) in zip(tm.beta_p, tm.beta_w):
            (S_ang, Z_ang) = tm.get_SZ_single(alpha=alpha, beta=beta)
            S += w * S_ang
            Z += w * Z_ang

    sw = tm.beta_w.sum()
    #normalize to get a proper average
    S *= aw/sw
    Z *= aw/sw

    return (S, Z)