def init_gaussian_hmm(observations, nstates, lag=1, reversible=True):
    """ Use a heuristic scheme to generate an initial model.

    Parameters
    ----------
    observations : list of ndarray((T_i))
        list of arrays of length T_i with observation data
    nstates : int
        The number of states.

    Examples
    --------

    Generate initial model for a gaussian output model.

    >>> import bhmm
    >>> [model, observations, states] = bhmm.testsystems.generate_synthetic_observations(output='gaussian')
    >>> initial_model = init_gaussian_hmm(observations, model.nstates)

    """
    from bhmm.init import gaussian
    if lag > 1:
        observations = lag_observations(observations, lag)
    hmm0 = gaussian.init_model_gaussian1d(observations, nstates, reversible=reversible)
    hmm0._lag = lag
    return hmm0