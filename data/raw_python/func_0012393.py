def parallel_map(func, *arg_iterable, **kwargs):
    """Apply function to iterable with parallel map, and hence returns
    results in order. functools.partial is used to freeze func_pre_args and
    func_kwargs, meaning that the iterable argument must be the last positional
    argument.

    Roughly equivalent to

    >>> [func(*func_pre_args, x, **func_kwargs) for x in arg_iterable]

    Parameters
    ----------
    func: function
        Function to apply to list of args.
    arg_iterable: iterable
        argument to iterate over.
    chunksize: int, optional
        Perform function in batches
    func_pre_args: tuple, optional
        Positional arguments to place before the iterable argument in func.
    func_kwargs: dict, optional
        Additional keyword arguments for func.
    parallel: bool, optional
        To turn off parallelisation if needed.
    parallel_warning: bool, optional
        To turn off warning for no parallelisation if needed.
    max_workers: int or None, optional
        Number of processes.
        If max_workers is None then concurrent.futures.ProcessPoolExecutor
        defaults to using the number of processors of the machine.
        N.B. If max_workers=None and running on supercomputer clusters with
        multiple nodes, this may default to the number of processors on a
        single node.

    Returns
    -------
    results_list: list of function outputs
    """
    chunksize = kwargs.pop('chunksize', 1)
    func_pre_args = kwargs.pop('func_pre_args', ())
    func_kwargs = kwargs.pop('func_kwargs', {})
    max_workers = kwargs.pop('max_workers', None)
    parallel = kwargs.pop('parallel', True)
    parallel_warning = kwargs.pop('parallel_warning', True)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    func_to_map = functools.partial(func, *func_pre_args, **func_kwargs)
    if parallel:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        return list(pool.map(func_to_map, *arg_iterable, chunksize=chunksize))
    else:
        if parallel_warning:
            warnings.warn(('parallel_map has parallel=False - turn on '
                           'parallelisation for faster processing'),
                          UserWarning)
        return list(map(func_to_map, *arg_iterable))