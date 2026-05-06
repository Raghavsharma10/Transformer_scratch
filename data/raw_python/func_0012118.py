def after(parents, func=None):
    '''Create a new Future whose completion depends on parent futures

    The new future will have a function that it calls once all its parents
    have completed, the return value of which will be its final value.
    There is a special case, however, in which the dependent future's
    callback returns a future or list of futures. In those cases, waiting
    on the dependent will also wait for all those futures, and the result
    (or list of results) of those future(s) will then be the final value.

    :param parents:
        A list of futures, all of which must be complete before the
        dependent's function runs.
    :type parents: list

    :param function func:
        The function to determine the value of the dependent future. It
        will take as many arguments as it has parents, and they will be the
        results of those futures.

    :returns:
        a :class:`Dependent`, which is a subclass of :class:`Future` and
        has all its capabilities.
    '''
    if func is None:
        return lambda f: after(parents, f)

    dep = Dependent(parents, func)
    for parent in parents:
        if parent.complete:
            dep._incoming(parent, parent.value)
        else:
            parent._children.append(weakref.ref(dep))
    return dep