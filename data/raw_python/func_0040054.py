def subclasses(self, sort_by=None, reverse=False):
        """Get all nested Constant class instance and it's name pair.

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

            >>> my_class = MyClass()
            >>> my_class.subclasses()
            [("C", my_class.C), ("D", my_class.D)]

        .. versionadded:: 0.0.4
        """
        l = list()
        for attr, _ in self.Subclasses(sort_by, reverse):
            value = getattr(self, attr)
            l.append((attr, value))
        return l