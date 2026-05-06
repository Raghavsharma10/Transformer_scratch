def build_and_trace(algo, init, limit=100, **kwargs):
    '''Run an optimizer on the rosenbrock function. Return xs, ys, and losses.

    In downhill, optimization algorithms can be iterated over to progressively
    minimize the loss. At each iteration, the optimizer yields a dictionary of
    monitor values that were computed during that iteration. Here we build an
    optimizer and then run it for a fixed number of iterations.
    '''
    kw = dict(min_improvement=0, patience=0, max_gradient_norm=100)
    kw.update(kwargs)
    xs, ys, loss = [], [], []
    for tm, _ in build(algo, init).iterate([[]], **kw):
        if len(init) == 2:
            xs.append(tm['x'])
            ys.append(tm['y'])
        loss.append(tm['loss'])
        if len(loss) == limit:
            break
    # Return the optimization up to any failure of patience.
    return xs[:-9], ys[:-9], loss[-9]