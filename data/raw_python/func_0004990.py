def PorodGuinierMulti(q, A, *alphasRgs):
    """Empirical multi-part Porod-Guinier scattering

    Inputs:
    -------
        ``q``: independent variable
        ``A``: factor for the first Power-law-branch
        other arguments: [alpha1, Rg1, alpha2, Rg2, alpha3 ...] the radii of
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
    scalefactor = A
    funcs = [lambda q: Powerlaw(q, A, alphasRgs[0])]
    indices = np.ones_like(q, dtype=np.bool)
    constraints = []
    for i in range(1, len(alphasRgs)):
        if i % 2:
            # alphasRgs[i] is a radius of gyration, alphasRgs[i-1] is a power-law exponent
            qsep = _PGgen_qsep(alphasRgs[i - 1], alphasRgs[i], 3)
            scalefactor = _PGgen_G(alphasRgs[i - 1], alphasRgs[i], 3, scalefactor)
            funcs.append(lambda q, G=scalefactor, Rg=alphasRgs[i]: Guinier(q, G, Rg))
        else:
            # alphasRgs[i] is an exponent, alphasRgs[i-1] is a radius of gyration
            qsep = _PGgen_qsep(alphasRgs[i], alphasRgs[i - 1], 3)
            scalefactor = _PGgen_A(alphasRgs[i], alphasRgs[i - 1], 3, scalefactor)
            funcs.append(lambda q, a=scalefactor, alpha=alphasRgs[i]: a * q ** alpha)
        # this belongs to the previous
        constraints.append(indices & (q < qsep))
        indices[q < qsep] = False
    constraints.append(indices)
    return np.piecewise(q, constraints, funcs)