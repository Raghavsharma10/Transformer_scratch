def Subclasses(cls, sort_by=None, reverse=False):
        """Get all nested Constant class and it's name pair.

        :param sort_by: the attribute name used for sorting.
        :param reverse: if True, return in descend order.
        :returns: [(attr, value),...] pairs.

        ::

        >>> class MyClass(Constant):
        ...     a = 1 # non-class attributre
        ...     b = 2 # non-class attributre
        ...
        ...     class C(Constant):
        ...         pass
        ...
        ...     class D(Constant):
        ...         pass

        >>> MyClass.Subclasses()
        [("C", MyClass.C), ("D", MyClass.D)]

        .. versionadded:: 0.0.3
        """
        l = list()
        for attr, value in get_all_attributes(cls):
            try:
                if issubclass(value, Constant):
                    l.append((attr, value))
            except:
                pass

        if sort_by is None:
            sort_by = "__creation_index__"

        l = list(
            sorted(l, key=lambda x: getattr(x[1], sort_by), reverse=reverse))

        return l