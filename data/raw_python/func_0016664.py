def encode_bqm_as_qp(solver, linear, quadratic):
    """Encode the binary quadratic problem for submission to a given solver,
    using the `qp` format for data.

    Args:
        solver (:class:`dwave.cloud.solver.Solver`):
            The solver used.

        linear (dict[variable, bias]/list[variable, bias]):
            Linear terms of the model.

        quadratic (dict[(variable, variable), bias]):
            Quadratic terms of the model.

    Returns:
        encoded submission dictionary
    """
    active = active_qubits(linear, quadratic)

    # Encode linear terms. The coefficients of the linear terms of the objective
    # are encoded as an array of little endian 64 bit doubles.
    # This array is then base64 encoded into a string safe for json.
    # The order of the terms is determined by the _encoding_qubits property
    # specified by the server.
    # Note: only active qubits are coded with double, inactive with NaN
    nan = float('nan')
    lin = [uniform_get(linear, qubit, 0 if qubit in active else nan)
           for qubit in solver._encoding_qubits]
    lin = base64.b64encode(struct.pack('<' + ('d' * len(lin)), *lin))

    # Encode the coefficients of the quadratic terms of the objective
    # in the same manner as the linear terms, in the order given by the
    # _encoding_couplers property, discarding tailing zero couplings
    quad = [quadratic.get((q1,q2), 0) + quadratic.get((q2,q1), 0)
            for (q1,q2) in solver._encoding_couplers
            if q1 in active and q2 in active]
    quad = base64.b64encode(struct.pack('<' + ('d' * len(quad)), *quad))

    # The name for this encoding is 'qp' and is explicitly included in the
    # message for easier extension in the future.
    return {
        'format': 'qp',
        'lin': lin.decode('utf-8'),
        'quad': quad.decode('utf-8')
    }