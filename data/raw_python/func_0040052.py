def items(self):
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

            >>> my_class = MyClass()
            >>> my_class.items()
            [("a", 1), ("b", 2)]

        .. versionchanged:: 0.0.5
        """
        l = list()
        # 为什么这里是 get_all_attributes(self.__class__) 而不是
        # get_all_attributes(self) ? 因为有些实例不支持
        # get_all_attributes(instance) 方法, 会报错。
        # 所以我们从类里得到所有的属性信息, 然后获得这些属性在实例中
        # 对应的值。
        for attr, value in get_all_attributes(self.__class__):
            value = getattr(self, attr)

            # if it is not a instance of class(Constant)
            if not isinstance(value, Constant):
                l.append((attr, value))

        return list(sorted(l, key=lambda x: x[0]))