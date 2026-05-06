def run_in_pool(target, iterable, threaded=True, processes=4,
                asynchronous=False, target_kwargs=None):
    """ Run a set of iterables to a function in a Threaded or MP Pool.

    .. code: python

        def func(a):
            return a + a

        reusables.run_in_pool(func, [1,2,3,4,5])
        # [1, 4, 9, 16, 25]


    :param target: function to run
    :param iterable: positional arg to pass to function
    :param threaded: Threaded if True multiprocessed if False
    :param processes: Number of workers
    :param asynchronous: will do map_async if True
    :param target_kwargs: Keyword arguments to set on the function as a partial
    :return: pool results
    """
    my_pool = pool.ThreadPool if threaded else pool.Pool

    if target_kwargs:
        target = partial(target, **target_kwargs if target_kwargs else None)

    p = my_pool(processes)
    try:
        results = (p.map_async(target, iterable) if asynchronous
                   else p.map(target, iterable))
    finally:
        p.close()
        p.join()
    return results