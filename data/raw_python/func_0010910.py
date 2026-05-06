def build(algo, init):
    '''Build and return an optimizer for the rosenbrock function.

    In downhill, an optimizer can be constructed using the build() top-level
    function. This function requires several Theano quantities such as the loss
    being optimized and the parameters to update during optimization.
    '''
    x = theano.shared(np.array(init, FLOAT), name='x')
    n = 0.1 * RandomStreams().normal((len(init) - 1, ))
    monitors = []
    if len(init) == 2:
        # this gives us access to the x and y locations during optimization.
        monitors.extend([('x', x[:-1].sum()), ('y', x[1:].sum())])
    return downhill.build(
        algo,
        loss=(n + 100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum(),
        params=[x],
        monitors=monitors,
        monitor_gradients=True)