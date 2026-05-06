def generate_measurements(party, label):
    """Generate variables that behave like measurements.

    :param party: The list of number of measurement outputs a party has.
    :type party: list of int.
    :param label: The label to be given to the symbolic variables.
    :type label: str.

    :returns: list of list of
             :class:`sympy.physics.quantum.operator.HermitianOperator`.
    """
    measurements = []
    for i in range(len(party)):
        measurements.append(generate_operators(label + '%s' % i, party[i] - 1,
                                               hermitian=True))
    return measurements