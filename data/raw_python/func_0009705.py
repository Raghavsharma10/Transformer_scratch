def cachier(stale_after=None, next_time=False, pickle_reload=True,
            mongetter=None):
    """A persistent, stale-free memoization decorator.

    The positional and keyword arguments to the wrapped function must be
    hashable (i.e. Python's immutable built-in objects, not mutable
    containers). Also, notice that since objects which are instances of
    user-defined classes are hashable but all compare unequal (their hash
    value is their id), equal objects across different sessions will not yield
    identical keys.

    Arguments
    ---------
    stale_after (optional) : datetime.timedelta
        The time delta afterwhich a cached result is considered stale. Calls
        made after the result goes stale will trigger a recalculation of the
        result, but whether a stale or fresh result will be returned is
        determined by the optional next_time argument.
    next_time (optional) : bool
        If set to True, a stale result will be returned when finding one, not
        waiting for the calculation of the fresh result to return. Defaults to
        False.
    pickle_reload (optional) : bool
        If set to True, in-memory cache will be reloaded on each cache read,
        enabling different threads to share cache. Should be set to False for
        faster reads in single-thread programs. Defaults to True.
    mongetter (optional) : callable
        A callable that takes no arguments and returns a pymongo.Collection
        object with writing permissions. If unset a local pickle cache is used
        instead.
    """
    # print('Inside the wrapper maker')
    # print('mongetter={}'.format(mongetter))
    # print('stale_after={}'.format(stale_after))
    # print('next_time={}'.format(next_time))

    if mongetter:
        core = _MongoCore(mongetter, stale_after, next_time)
    else:
        core = _PickleCore(  # pylint: disable=R0204
            stale_after, next_time, pickle_reload)

    def _cachier_decorator(func):
        core.set_func(func)

        @wraps(func)
        def func_wrapper(*args, **kwds):  # pylint: disable=C0111,R0911
            # print('Inside general wrapper for {}.'.format(func.__name__))
            ignore_cache = kwds.pop('ignore_cache', False)
            overwrite_cache = kwds.pop('overwrite_cache', False)
            verbose_cache = kwds.pop('verbose_cache', False)
            _print = lambda x: None
            if verbose_cache:
                _print = print
            if ignore_cache:
                return func(*args, **kwds)
            key, entry = core.get_entry(args, kwds)
            if overwrite_cache:
                return _calc_entry(core, key, func, args, kwds)
            if entry is not None:  # pylint: disable=R0101
                _print('Entry found.')
                if entry.get('value', None) is not None:
                    _print('Cached result found.')
                    if stale_after:
                        now = datetime.datetime.now()
                        if now - entry['time'] > stale_after:
                            _print('But it is stale... :(')
                            if entry['being_calculated']:
                                if next_time:
                                    _print('Returning stale.')
                                    return entry['value']  # return stale val
                                _print('Already calc. Waiting on change.')
                                try:
                                    return core.wait_on_entry_calc(key)
                                except RecalculationNeeded:
                                    return _calc_entry(core, key, func, args, kwds)
                            if next_time:
                                _print('Async calc and return stale')
                                try:
                                    core.mark_entry_being_calculated(key)
                                    _get_executor().submit(
                                        _function_thread, core, key, func,
                                        args, kwds)
                                finally:
                                    core.mark_entry_not_calculated(key)
                                return entry['value']
                            _print('Calling decorated function and waiting')
                            return _calc_entry(core, key, func, args, kwds)
                    _print('And it is fresh!')
                    return entry['value']
                if entry['being_calculated']:
                    _print('No value but being calculated. Waiting.')
                    try:
                        return core.wait_on_entry_calc(key)
                    except RecalculationNeeded:
                        return _calc_entry(core, key, func, args, kwds)
            _print('No entry found. No current calc. Calling like a boss.')
            return _calc_entry(core, key, func, args, kwds)

        def clear_cache():
            """Clear the cache."""
            core.clear_cache()

        def clear_being_calculated():
            """Marks all entries in this cache as not being calculated."""
            core.clear_being_calculated()

        func_wrapper.clear_cache = clear_cache
        func_wrapper.clear_being_calculated = clear_being_calculated
        return func_wrapper

    return _cachier_decorator