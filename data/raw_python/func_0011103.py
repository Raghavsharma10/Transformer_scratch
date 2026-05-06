def projective_measurement_constraints(*parties):
    """Return a set of constraints that define projective measurements.

    :param parties: Measurements of different parties.
    :type A: list or tuple of list of list of
             :class:`sympy.physics.quantum.operator.HermitianOperator`.

    :returns: substitutions containing idempotency, orthogonality and
              commutation relations.
    """
    substitutions = {}
    # Idempotency and orthogonality of projectors
    if isinstance(parties[0][0][0], list):
        parties = parties[0]
    for party in parties:
        for measurement in party:
            for projector1 in measurement:
                for projector2 in measurement:
                    if projector1 == projector2:
                        substitutions[projector1**2] = projector1
                    else:
                        substitutions[projector1*projector2] = 0
                        substitutions[projector2*projector1] = 0
    # Projectors commute between parties in a partition
    for n1 in range(len(parties)):
        for n2 in range(n1+1, len(parties)):
            for measurement1 in parties[n1]:
                for measurement2 in parties[n2]:
                    for projector1 in measurement1:
                        for projector2 in measurement2:
                            substitutions[projector2*projector1] = \
                                projector1*projector2
    return substitutions