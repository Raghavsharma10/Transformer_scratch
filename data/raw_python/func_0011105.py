def correlator(A, B):
    """Correlators between the probabilities of two parties.

    :param A: Measurements of Alice.
    :type A: list of list of
             :class:`sympy.physics.quantum.operator.HermitianOperator`.
    :param B: Measurements of Bob.
    :type B: list of list of
             :class:`sympy.physics.quantum.operator.HermitianOperator`.

    :returns: list of correlators.
    """
    correlators = []
    for i in range(len(A)):
        correlator_row = []
        for j in range(len(B)):
            corr = 0
            for k in range(len(A[i])):
                for l in range(len(B[j])):
                    if k == l:
                        corr += A[i][k] * B[j][l]
                    else:
                        corr -= A[i][k] * B[j][l]
            correlator_row.append(corr)
        correlators.append(correlator_row)
    return correlators