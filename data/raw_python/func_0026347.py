def _ensure_coroutine_function(func):
    """Return a coroutine function.

    func: either a coroutine function or a regular function

    Note a coroutine function is not a coroutine!
    """
    if asyncio.iscoroutinefunction(func):
        return func
    else:
        @asyncio.coroutine
        def coroutine_function(evt):
            func(evt)
            yield
        return coroutine_function