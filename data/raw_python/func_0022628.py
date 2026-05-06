def implementation(self, for_type=None, for_types=None):
        """Return a decorator that will register the implementation.

        Example:
            @multimethod
            def add(x, y):
                pass

            @add.implementation(for_type=int)
            def add(x, y):
                return x + y

            @add.implementation(for_type=SomeType)
            def add(x, y):
                return int(x) + int(y)
        """
        for_types = self.__get_types(for_type, for_types)

        def _decorator(implementation):
            self.implement(implementation, for_types=for_types)
            return self

        return _decorator