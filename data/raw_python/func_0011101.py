def pauli_constraints(X, Y, Z):
    """Return  a set of constraints that define Pauli spin operators.

    :param X: List of Pauli X operator on sites.
    :type X: list of :class:`sympy.physics.quantum.operator.HermitianOperator`.
    :param Y: List of Pauli Y operator on sites.
    :type Y: list of :class:`sympy.physics.quantum.operator.HermitianOperator`.
    :param Z: List of Pauli Z operator on sites.
    :type Z: list of :class:`sympy.physics.quantum.operator.HermitianOperator`.

    :returns: tuple of substitutions and equalities.
    """
    substitutions = {}
    n_vars = len(X)
    for i in range(n_vars):
        # They square to the identity
        substitutions[X[i] * X[i]] = 1
        substitutions[Y[i] * Y[i]] = 1
        substitutions[Z[i] * Z[i]] = 1

        # Anticommutation relations
        substitutions[Y[i] * X[i]] = - X[i] * Y[i]
        substitutions[Z[i] * X[i]] = - X[i] * Z[i]
        substitutions[Z[i] * Y[i]] = - Y[i] * Z[i]
        # Commutation relations.
        # equalities.append(X[i]*Y[i] - 1j*Z[i])
        # equalities.append(X[i]*Z[i] + 1j*Y[i])
        # equalities.append(Y[i]*Z[i] - 1j*X[i])
        # They commute between the sites
        for j in range(i + 1, n_vars):
            substitutions[X[j] * X[i]] = X[i] * X[j]
            substitutions[Y[j] * Y[i]] = Y[i] * Y[j]
            substitutions[Y[j] * X[i]] = X[i] * Y[j]
            substitutions[Y[i] * X[j]] = X[j] * Y[i]
            substitutions[Z[j] * Z[i]] = Z[i] * Z[j]
            substitutions[Z[j] * X[i]] = X[i] * Z[j]
            substitutions[Z[i] * X[j]] = X[j] * Z[i]
            substitutions[Z[j] * Y[i]] = Y[i] * Z[j]
            substitutions[Z[i] * Y[j]] = Y[j] * Z[i]
    return substitutions