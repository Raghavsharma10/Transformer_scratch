def memoized(fn=None, cache=None):
    """ Memoize a function into an optionally-specificed cache container.

    If the `cache` container is not specified, then the instance container is
    accessible from the wrapped function's `memoize_cache` property.

    Example::

        >>> @memoized
        ... def foo(bar):
        ...   print("Not cached.")
        >>> foo(1)
        Not cached.
        >>> foo(1)
        >>> foo(2)
        Not cached.

    Example with a specific cache container (in this case, the
    ``RecentlyUsedContainer``, which will only store the ``maxsize`` most
    recently accessed items)::

        >>> from unstdlib.standard.collections_ import RecentlyUsedContainer
        >>> lru_container = RecentlyUsedContainer(maxsize=2)
        >>> @memoized(cache=lru_container)
        ... def baz(x):
        ...   print("Not cached.")
        >>> baz(1)
        Not cached.
        >>> baz(1)
        >>> baz(2)
        Not cached.
        >>> baz(3)
        Not cached.
        >>> baz(2)
        >>> baz(1)
        Not cached.
        >>> # Notice that the '2' key remains, but the '1' key was evicted from
        >>> # the cache.
    """
    if fn:
        # This is a hack to support both @memoize and @memoize(...)
        return memoized(cache=cache)(fn)

    if cache is None:
        cache = {}

    def decorator(fn):
        wrapped = wraps(fn)(partial(_memoized_call, fn, cache))
        wrapped.memoize_cache = cache
        return wrapped

    return decorator