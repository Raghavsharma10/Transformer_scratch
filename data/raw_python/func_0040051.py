def Items(cls):
        """non-class attributes ordered by alphabetical order.

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

            >>> MyClass.Items()
            [("a", 1), ("b", 2)]

        .. versionadded:: 0.0.5
        """
        l = list()
        for attr, value in get_all_attributes(cls):
            # if it's not a class(Constant)
            if not inspect.isclass(value):
                l.append((attr, value))

        return list(sorted(l, key=lambda x: x[0]))