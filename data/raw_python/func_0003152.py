def orient_averaged_adaptive(tm):
    """Compute the T-matrix using variable orientation scatterers.
    
    This method uses a very slow adaptive routine and should mainly be used
    for reference purposes. Uses the set particle orientation PDF, ignoring
    the alpha and beta attributes.

    Args:
        tm: TMatrix (or descendant) instance

    Returns:
        The amplitude (S) and phase (Z) matrices.
    """
    S = np.zeros((2,2), dtype=complex)
    Z = np.zeros((4,4))

    def Sfunc(beta, alpha, i, j, real):
        (S_ang, Z_ang) = tm.get_SZ_single(alpha=alpha, beta=beta)
        s = S_ang[i,j].real if real else S_ang[i,j].imag            
        return s * tm.or_pdf(beta)

    ind = range(2)
    for i in ind:
        for j in ind:
            S.real[i,j] = dblquad(Sfunc, 0.0, 360.0, 
                lambda x: 0.0, lambda x: 180.0, (i,j,True))[0]/360.0        
            S.imag[i,j] = dblquad(Sfunc, 0.0, 360.0, 
                lambda x: 0.0, lambda x: 180.0, (i,j,False))[0]/360.0

    def Zfunc(beta, alpha, i, j):
        (S_and, Z_ang) = tm.get_SZ_single(alpha=alpha, beta=beta)
        return Z_ang[i,j] * tm.or_pdf(beta)

    ind = range(4)
    for i in ind:
        for j in ind:
            Z[i,j] = dblquad(Zfunc, 0.0, 360.0, 
                lambda x: 0.0, lambda x: 180.0, (i,j))[0]/360.0

    return (S, Z)