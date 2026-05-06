def active_qubits(linear, quadratic):
    """Calculate a set of all active qubits. Qubit is "active" if it has
    bias or coupling attached.

    Args:
        linear (dict[variable, bias]/list[variable, bias]):
            Linear terms of the model.

        quadratic (dict[(variable, variable), bias]):
            Quadratic terms of the model.

    Returns:
        set:
            Active qubits' indices.
    """

    active = {idx for idx,bias in uniform_iterator(linear)}
    for edge, _ in six.iteritems(quadratic):
        active.update(edge)
    return active