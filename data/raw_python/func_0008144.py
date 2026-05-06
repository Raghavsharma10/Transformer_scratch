def init_hmm(observations, nstates, lag=1, output=None, reversible=True):
    """Use a heuristic scheme to generate an initial model.

    Parameters
    ----------
    observations : list of ndarray((T_i))
        list of arrays of length T_i with observation data
    nstates : int
        The number of states.
    output : str, optional, default=None
        Output model type from [None, 'gaussian', 'discrete']. If None, will automatically select an output
        model type based on the format of observations.

    Examples
    --------

    Generate initial model for a gaussian output model.

    >>> import bhmm
    >>> [model, observations, states] = bhmm.testsystems.generate_synthetic_observations(output='gaussian')
    >>> initial_model = init_hmm(observations, model.nstates, output='gaussian')

    Generate initial model for a discrete output model.

    >>> import bhmm
    >>> [model, observations, states] = bhmm.testsystems.generate_synthetic_observations(output='discrete')
    >>> initial_model = init_hmm(observations, model.nstates, output='discrete')

    """
    # select output model type
    if output is None:
        output = _guess_output_type(observations)

    if output == 'discrete':
        return init_discrete_hmm(observations, nstates, lag=lag, reversible=reversible)
    elif output == 'gaussian':
        return init_gaussian_hmm(observations, nstates, lag=lag, reversible=reversible)
    else:
        raise NotImplementedError('output model type '+str(output)+' not yet implemented.')