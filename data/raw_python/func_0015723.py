def register(cls, namespace, name):
        """Class decorator"""

        def func(kind):
            cls._FOREIGN[(namespace, name)] = kind()
            return kind
        return func