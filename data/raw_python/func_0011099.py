def bosonic_constraints(a):
    """Return  a set of constraints that define fermionic ladder operators.

    :param a: The non-Hermitian variables.
    :type a: list of :class:`sympy.physics.quantum.operator.Operator`.
    :returns: a dict of substitutions.
    """
    substitutions = {}
    for i, ai in enumerate(a):
        substitutions[ai * Dagger(ai)] = 1.0 + Dagger(ai) * ai
        for aj in a[i+1:]:
            # substitutions[ai*Dagger(aj)] = -Dagger(ai)*aj
            substitutions[ai*Dagger(aj)] = Dagger(aj)*ai
            substitutions[Dagger(ai)*aj] = aj*Dagger(ai)
            substitutions[ai*aj] = aj*ai
            substitutions[Dagger(ai) * Dagger(aj)] = Dagger(aj) * Dagger(ai)

    return substitutions