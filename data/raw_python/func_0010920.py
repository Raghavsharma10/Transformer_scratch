def build(algo, loss, params=None, inputs=None, updates=(), monitors=(),
          monitor_gradients=False):
    '''Construct an optimizer by name.

    Parameters
    ----------
    algo : str
        The name of the optimization algorithm to build.
    loss : Theano expression
        Loss function to minimize. This must be a scalar-valued expression.
    params : list of Theano variables, optional
        Symbolic variables to adjust to minimize the loss. If not given, these
        will be computed automatically by walking the computation graph.
    inputs : list of Theano variables, optional
        Symbolic variables required to compute the loss. If not given, these
        will be computed automatically by walking the computation graph.
    updates : list of update pairs, optional
        A list of pairs providing updates for the internal of the loss
        computation. Normally this is empty, but it can be provided if the loss,
        for example, requires an update to an internal random number generator.
    monitors : dict or sequence of (str, Theano expression) tuples, optional
        Additional values to monitor during optimization. These must be provided
        as either a sequence of (name, expression) tuples, or as a dictionary
        mapping string names to Theano expressions.
    monitor_gradients : bool, optional
        If True, add monitors to log the norms of the parameter gradients during
        optimization. Defaults to False.

    Returns
    -------
    optimizer : :class:`Optimizer`
        An optimizer instance.
    '''
    return Optimizer.build(algo, loss, params, inputs,
                           updates=updates, monitors=monitors,
                           monitor_gradients=monitor_gradients)