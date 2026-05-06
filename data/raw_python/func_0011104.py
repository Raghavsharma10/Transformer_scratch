def define_objective_with_I(I, *args):
    """Define a polynomial using measurements and an I matrix describing a Bell
    inequality.

    :param I: The I matrix of a Bell inequality in the Collins-Gisin notation.
    :type I: list of list of int.
    :param args: Either the measurements of Alice and Bob or a `Probability`
                 class describing their measurement operators.
    :type A: tuple of list of list of
             :class:`sympy.physics.quantum.operator.HermitianOperator` or
             :class:`ncpol2sdpa.Probability`

    :returns: :class:`sympy.core.expr.Expr` -- the objective function to be
              solved as a minimization problem to find the maximum quantum
              violation. Note that the sign is flipped compared to the Bell
              inequality.
    """
    objective = I[0][0]
    if len(args) > 2 or len(args) == 0:
        raise Exception("Wrong number of arguments!")
    elif len(args) == 1:
        A = args[0].parties[0]
        B = args[0].parties[1]
    else:
        A = args[0]
        B = args[1]
    i, j = 0, 1  # Row and column index in I
    for m_Bj in B:  # Define first row
        for Bj in m_Bj:
            objective += I[i][j] * Bj
            j += 1
    i += 1
    for m_Ai in A:
        for Ai in m_Ai:
            objective += I[i][0] * Ai
            j = 1
            for m_Bj in B:
                for Bj in m_Bj:
                    objective += I[i][j] * Ai * Bj
                    j += 1
            i += 1
    return -objective