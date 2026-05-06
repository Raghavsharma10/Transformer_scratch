def evaluate_ising(linear, quad, state):
    """Calculate the energy of a state given the Hamiltonian.

    Args:
        linear: Linear Hamiltonian terms.
        quad: Quadratic Hamiltonian terms.
        state: Vector of spins describing the system state.

    Returns:
        Energy of the state evaluated by the given energy function.
    """

    # If we were given a numpy array cast to list
    if _numpy and isinstance(state, np.ndarray):
        return evaluate_ising(linear, quad, state.tolist())

    # Accumulate the linear and quadratic values
    energy = 0.0
    for index, value in uniform_iterator(linear):
        energy += state[index] * value
    for (index_a, index_b), value in six.iteritems(quad):
        energy += value * state[index_a] * state[index_b]
    return energy