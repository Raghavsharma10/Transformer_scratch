def _pairwise_imp(function, x, y=None, pool=None, is_symmetric=None, **kwargs):
    """
    Real implementation of :func:`pairwise`.

    This function is used to make several parameters keyword-only in
    Python 2.

    """
    map_function = pool.map if pool else map

    if is_symmetric is None:
        is_symmetric = getattr(function, 'is_symmetric', False)

    pairwise_function = getattr(function, 'pairwise_function', None)
    if pairwise_function:
        return pairwise_function(x, y, pool=pool, is_symmetric=is_symmetric,
                                 **kwargs)

    if y is None and is_symmetric:

        partial = functools.partial(_map_aux_func_symmetric, x=x,
                                    function=function)

        dependencies = np.array(list(map_function(partial, enumerate(x))))

        for i in range(len(x)):
            for j in range(i, len(x)):
                dependencies[j, i] = dependencies[i, j]

        return dependencies

    else:
        if y is None:
            y = x

        partial = functools.partial(_map_aux_func, y=y, function=function)

        return np.array(list(map_function(partial, x)))