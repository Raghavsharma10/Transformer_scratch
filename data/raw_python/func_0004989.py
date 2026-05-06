def GuinierPorodMulti(q, G, *Rgsalphas):
    """Empirical multi-part Guinier-Porod scattering

    Inputs:
    -------
        ``q``: independent variable
        ``G``: factor for the first Guinier-branch
        other arguments: [Rg1, alpha1, Rg2, alpha2, Rg3 ...] the radii of
        gyration and power-law exponents of the consecutive parts

    Formula:
    --------
        The intensity is a piecewise function with continuous first derivatives.
        The separating points in ``q`` between the consecutive parts and the
        intensity factors of them (except the first) are determined from
        conditions of smoothness (continuity of the function and its first
        derivative) at the border points of the intervals. Guinier-type
        (``G*exp(-q^2*Rg1^2/3)``) and Power-law type (``A*q^alpha``) parts
        follow each other in alternating sequence.

    Literature:
    -----------
        B. Hammouda: A new Guinier-Porod model. J. Appl. Crystallogr. (2010) 43,
            716-719.
    """
    scalefactor = G
    funcs = [lambda q: Guinier(q, G, Rgsalphas[0])]
    indices = np.ones_like(q, dtype=np.bool)
    constraints = []
    for i in range(1, len(Rgsalphas)):
        if i % 2:
            # Rgsalphas[i] is an exponent, Rgsalphas[i-1] is a radius of gyration
            qsep = _PGgen_qsep(Rgsalphas[i], Rgsalphas[i - 1], 3)
            scalefactor = _PGgen_A(Rgsalphas[i], Rgsalphas[i - 1], 3, scalefactor)
            funcs.append(lambda q, a=scalefactor, alpha=Rgsalphas[i]: Powerlaw(q, a, alpha))
        else:
            # Rgsalphas[i] is a radius of gyration, Rgsalphas[i-1] is a power-law exponent
            qsep = _PGgen_qsep(Rgsalphas[i - 1], Rgsalphas[i], 3)
            scalefactor = _PGgen_G(Rgsalphas[i - 1], Rgsalphas[i], 3, scalefactor)
            funcs.append(lambda q, G=scalefactor, Rg=Rgsalphas[i]: Guinier(q, G, Rg))
        # this belongs to the previous
        constraints.append(indices & (q < qsep))
        indices[q < qsep] = False
    constraints.append(indices)
    return np.piecewise(q, constraints, funcs)