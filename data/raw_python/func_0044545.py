def recursive_iterator(func):
    """Decorates a function by optimizing it for iterator recursion.
    Requires function arguments to be pickleable."""
    tee_store = {}
    @_coconut.functools.wraps(func)
    def recursive_iterator_func(*args, **kwargs):
        hashable_args_kwargs = _coconut.pickle.dumps((args, kwargs), _coconut.pickle.HIGHEST_PROTOCOL)
        try:
            to_tee = tee_store[hashable_args_kwargs]
        except _coconut.KeyError:
            to_tee = func(*args, **kwargs)
        tee_store[hashable_args_kwargs], to_return = _coconut_tee(to_tee)
        return to_return
    return recursive_iterator_func