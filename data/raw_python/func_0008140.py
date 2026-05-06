def _guess_output_type(observations):
    """ Suggests a HMM model type based on the observation data

    Uses simple rules in order to decide which HMM model type makes sense based on observation data.
    If observations consist of arrays/lists of integer numbers (irrespective of whether the python type is
    int or float), our guess is 'discrete'.
    If observations consist of arrays/lists of 1D-floats, our guess is 'discrete'.
    In any other case, a TypeError is raised because we are not supporting that data type yet.

    Parameters
    ----------
    observations : list of lists or arrays
        observation trajectories

    Returns
    -------
    output_type : str
        One of {'discrete', 'gaussian'}

    """
    from bhmm.util import types as _types

    o1 = _np.array(observations[0])

    # CASE: vector of int? Then we want a discrete HMM
    if _types.is_int_vector(o1):
        return 'discrete'

    # CASE: not int type, but everything is an integral number. Then we also go for discrete
    if _np.allclose(o1, _np.round(o1)):
        isintegral = True
        for i in range(1, len(observations)):
            if not _np.allclose(observations[i], _np.round(observations[i])):
                isintegral = False
                break
        if isintegral:
            return 'discrete'

    # CASE: vector of double? Then we want a gaussian
    if _types.is_float_vector(o1):
        return 'gaussian'

    # None of the above? Then we currently do not support this format!
    raise TypeError('Observations is neither sequences of integers nor 1D-sequences of floats. The current version'
                    'does not support your input.')