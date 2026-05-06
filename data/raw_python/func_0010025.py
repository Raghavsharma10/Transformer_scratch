def __itersubclasses(cls, _seen=None):
        """
        
        Credit goes to: http://code.activestate.com/recipes/576949-find-all-subclasses-of-a-given-class/

        itersubclasses(cls)

        Generator over all subclasses of a given class, in depth first order.

        >>> list(itersubclasses(int)) == [bool]
        True
        >>> class A(object): pass
        >>> class B(A): pass
        >>> class C(A): pass
        >>> class D(B,C): pass
        >>> class E(D): pass
        >>> 
        >>> for cls in itersubclasses(A):
        ...     print(cls.__name__)
        B
        D
        E
        C
        >>> # get ALL (new-style) classes currently defined
        >>> [cls.__name__ for cls in itersubclasses(object)] #doctest: +ELLIPSIS
        ['type', ...'tuple', ...]
        """
        if not isinstance(cls, type):
            raise TypeError(u('Argument ({0}) passed to PageFactory does not appear to be a valid Class.').format(cls),
                            "Check to make sure the first parameter is an PageObject class, interface, or mixin.")
        if _seen is None:
            _seen = set()
        try:
            subs = cls.__subclasses__()
        except TypeError:  # fails only when cls is type
            subs = cls.__subclasses__(cls)
        for sub in subs:
            if sub not in _seen:
                _seen.add(sub)
                yield sub
                for sub in PageFactory.__itersubclasses(sub, _seen):
                    yield sub