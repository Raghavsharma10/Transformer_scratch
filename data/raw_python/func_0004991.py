def GeneralGuinierPorod(q, factor, *args, **kwargs):
    """Empirical generalized multi-part Guinier-Porod scattering

    Inputs:
    -------
        ``q``: independent variable
        ``factor``: factor for the first branch
        other arguments (*args): the defining arguments of the consecutive
             parts: radius of gyration (``Rg``) and dimensionality
             parameter (``s``) for Guinier and exponent (``alpha``) for
             power-law parts.
        supported keyword arguments:
            ``startswithguinier``: True if the first segment is a Guinier-type
            scattering (this is the default) or False if it is a power-law

    Formula:
    --------
        The intensity is a piecewise function with continuous first derivatives.
        The separating points in ``q`` between the consecutive parts and the
        intensity factors of them (except the first) are determined from
        conditions of smoothness (continuity of the function and its first
        derivative) at the border points of the intervals. Guinier-type
        (``G*q**(3-s)*exp(-q^2*Rg1^2/s)``) and Power-law type (``A*q^alpha``)
        parts follow each other in alternating sequence. The exact number of
        parts is determined from the number of positional arguments (*args).

    Literature:
    -----------
        B. Hammouda: A new Guinier-Porod model. J. Appl. Crystallogr. (2010) 43,
            716-719.
    """
    if kwargs.get('startswithguinier', True):
        funcs = [lambda q, A = factor:GeneralGuinier(q, A, args[0], args[1])]
        i = 2
        guiniernext = False
    else:
        funcs = [lambda q, A = factor: Powerlaw(q, A, args[0])]
        i = 1
        guiniernext = True
    indices = np.ones_like(q, dtype=np.bool)
    constraints = []
    while i < len(args):
        if guiniernext:
            # args[i] is a radius of gyration, args[i+1] is a dimensionality parameter, args[i-1] is a power-law exponent
            qsep = _PGgen_qsep(args[i - 1], args[i], args[i + 1])
            factor = _PGgen_G(args[i - 1], args[i], args[i + 1], factor)
            funcs.append(lambda q, G=factor, Rg=args[i], s=args[i + 1]: GeneralGuinier(q, G, Rg, s))
            guiniernext = False
            i += 2
        else:
            # args[i] is an exponent, args[i-2] is a radius of gyration, args[i-1] is a dimensionality parameter
            qsep = _PGgen_qsep(args[i], args[i - 2], args[i - 1])
            factor = _PGgen_A(args[i], args[i - 2], args[i - 1], factor)
            funcs.append(lambda q, a=factor, alpha=args[i]: a * q ** alpha)
            guiniernext = True
            i += 1
        # this belongs to the previous
        constraints.append(indices & (q < qsep))
        indices[q < qsep] = False
    constraints.append(indices)
    return np.piecewise(q, constraints, funcs)