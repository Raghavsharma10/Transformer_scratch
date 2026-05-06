def skip_duplicates(iterable, key=None, fingerprints=()):
    # type: (Iterable, Callable, Any) -> Iterable
    """
        Returns a generator that will yield all objects from iterable, skipping
        duplicates.

        Duplicates are identified using the `key` function to calculate a
        unique fingerprint. This does not use natural equality, but the
        result use a set() to remove duplicates, so defining __eq__
        on your objects would have no effect.

        By default the fingerprint is the object itself,
        which ensure the functions works as-is with an iterable of primitives
        such as int, str or tuple.

        :Example:

            >>> list(skip_duplicates([1, 2, 3, 4, 4, 2, 1, 3 , 4]))
            [1, 2, 3, 4]

        The return value of `key` MUST be hashable, which means for
        non hashable objects such as dict, set or list, you need to specify
        a a function that returns a hashable fingerprint.

        :Example:

            >>> list(skip_duplicates(([], [], (), [1, 2], (1, 2)),
            ...                      lambda x: tuple(x)))
            [[], [1, 2]]
            >>> list(skip_duplicates(([], [], (), [1, 2], (1, 2)),
            ...                      lambda x: (type(x), tuple(x))))
            [[], (), [1, 2], (1, 2)]

        For more complex types, such as custom classes, the default behavior
        is to remove nothing. You MUST provide a `key` function is you wish
        to filter those.

        :Example:

            >>> class Test(object):
            ...    def __init__(self, foo='bar'):
            ...        self.foo = foo
            ...    def __repr__(self):
            ...        return "Test('%s')" % self.foo
            ...
            >>> list(skip_duplicates([Test(), Test(), Test('other')]))
            [Test('bar'), Test('bar'), Test('other')]
            >>> list(skip_duplicates([Test(), Test(), Test('other')],\
                                     lambda x: x.foo))
            [Test('bar'), Test('other')]

    """

    fingerprints = fingerprints or set()
    fingerprint = None  # needed on type errors unrelated to hashing

    try:
        # duplicate some code to gain perf in the most common case
        if key is None:
            for x in iterable:
                if x not in fingerprints:
                    yield x
                    fingerprints.add(x)
        else:
            for x in iterable:
                fingerprint = key(x)
                if fingerprint not in fingerprints:
                    yield x
                    fingerprints.add(fingerprint)
    except TypeError:
        try:
            hash(fingerprint)
        except TypeError:
            raise TypeError(
                "The 'key' function returned a non hashable object of type "
                "'%s' when receiving '%s'. Make sure this function always "
                "returns a hashable object. Hint: immutable primitives like"
                "int, str or tuple, are hashable while dict, set and list are "
                "not." % (type(fingerprint), x))
        else:
            raise