def memoized_method(method=None, cache_factory=None):
    """ Memoize a class's method.

    Arguments are similar to to `memoized`, except that the cache container is
    specified with `cache_factory`: a function called with no arguments to
    create the caching container for the instance.

    Note that, unlike `memoized`, the result cache will be stored on the
    instance, so cached results will be deallocated along with the instance.

    Example::

        >>> class Person(object):
        ...     def __init__(self, name):
        ...         self._name = name
        ...     @memoized_method
        ...     def get_name(self):
        ...         print("Calling get_name on %r" %(self._name, ))
        ...         return self._name
        >>> shazow = Person("shazow")
        >>> shazow.get_name()
        Calling get_name on 'shazow'
        'shazow'
        >>> shazow.get_name()
        'shazow'
        >>> shazow._get_name_cache
        {((), ()): 'shazow'}

    Example with a specific cache container::

        >>> from unstdlib.standard.collections_ import RecentlyUsedContainer
        >>> class Foo(object):
        ...     @memoized_method(cache_factory=lambda: RecentlyUsedContainer(maxsize=2))
        ...     def add(self, a, b):
        ...         print("Calling add with %r and %r" %(a, b))
        ...         return a + b
        >>> foo = Foo()
        >>> foo.add(1, 1)
        Calling add with 1 and 1
        2
        >>> foo.add(1, 1)
        2
        >>> foo.add(2, 2)
        Calling add with 2 and 2
        4
        >>> foo.add(3, 3)
        Calling add with 3 and 3
        6
        >>> foo.add(1, 1)
        Calling add with 1 and 1
        2
    """

    if method is None:
        return lambda f: memoized_method(f, cache_factory=cache_factory)

    cache_factory = cache_factory or dict

    @wraps(method)
    def memoized_method_property(self):
        cache = cache_factory()
        cache_attr = "_%s_cache" %(method.__name__, )
        setattr(self, cache_attr, cache)
        result = partial(
            _memoized_call,
            partial(method, self),
            cache
        )
        result.memoize_cache = cache
        return result
    return memoized_property(memoized_method_property)