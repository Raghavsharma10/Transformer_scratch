def parallel_apply(func, arg_iterable, **kwargs):
    """Apply function to iterable with parallelisation and a tqdm progress bar.

    Roughly equivalent to

    >>> [func(*func_pre_args, x, *func_args, **func_kwargs) for x in
         arg_iterable]

    but will **not** necessarily return results in input order.

    Parameters
    ----------
    func: function
        Function to apply to list of args.
    arg_iterable: iterable
        argument to iterate over.
    func_args: tuple, optional
        Additional positional arguments for func.
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
    max_workers = kwargs.pop('max_workers', None)
    parallel = kwargs.pop('parallel', True)
    parallel_warning = kwargs.pop('parallel_warning', True)
    func_args = kwargs.pop('func_args', ())
    func_pre_args = kwargs.pop('func_pre_args', ())
    func_kwargs = kwargs.pop('func_kwargs', {})
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {})
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    if 'leave' not in tqdm_kwargs:  # default to leave=False
        tqdm_kwargs['leave'] = False
    assert isinstance(func_args, tuple), (
        str(func_args) + ' is type ' + str(type(func_args)))
    assert isinstance(func_pre_args, tuple), (
        str(func_pre_args) + ' is type ' + str(type(func_pre_args)))
    progress = select_tqdm()
    if not parallel:
        if parallel_warning:
            warnings.warn(('parallel_map has parallel=False - turn on '
                           'parallelisation for faster processing'),
                          UserWarning)
        return [func(*(func_pre_args + (x,) + func_args), **func_kwargs) for
                x in progress(arg_iterable, **tqdm_kwargs)]
    else:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        futures = []
        for element in arg_iterable:
            futures.append(pool.submit(
                func, *(func_pre_args + (element,) + func_args),
                **func_kwargs))
        results = []
        for fut in progress(concurrent.futures.as_completed(futures),
                            total=len(arg_iterable), **tqdm_kwargs):
            results.append(fut.result())
        return results