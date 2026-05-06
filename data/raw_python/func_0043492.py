def pmap(func, args, processes=None, callback=lambda *_, **__: None, **kwargs):
    """pmap(func, args, processes=None, callback=do_nothing, **kwargs)

    Parallel equivalent of ``map(func, args)``, with the additional ability of
    providing keyword arguments to func, and a callback function which is
    applied to each element in the returned list. Unlike map, the output is a
    non-lazy list. If *processes* is 1, no thread pool is used.

    **Parameters**

    func : function
        The function to map.
    args : iterable
        The arguments to map *func* over.
    processes : int or None, optional
        The number of processes in the thread pool. If only 1, no thread pool
        is used to avoid useless overhead. If None, the number is chosen based
        on your system by :class:`multiprocessing.Pool` (default None).
    callback : function, optional
        Function to call on the return value of ``func(arg)`` for each *arg*
        in *args* (default do_nothing).
    kwargs : dict
        Extra keyword arguments are unpacked in each call of *func*.

    **Returns**

    results : list
        A list equivalent to ``[func(x, **kwargs) for x in args]``.
    """
    if processes is 1:
        results = []
        for arg in args:
            result = func(arg, **kwargs)
            results.append(result)
            callback(result)

        return results
    else:
        with Pool() if processes is None else Pool(processes) as p:
            results = [p.apply_async(func, (arg,), kwargs, callback)
                       for arg in args]

            return [result.get() for result in results]