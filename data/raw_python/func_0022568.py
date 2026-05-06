def implemented(cls, for_type):
        """Assert that protocol 'cls' is implemented for type 'for_type'.

        This will cause 'for_type' to be registered with the protocol 'cls'.
        Subsequently, protocol.isa(for_type, cls) will return True, as will
        isinstance, issubclass and others.

        Raises:
            TypeError if 'for_type' doesn't implement all required functions.
        """

        for function in cls.required():
            if not function.implemented_for_type(for_type):
                raise TypeError(
                    "%r doesn't implement %r so it cannot participate in "
                    "the protocol %r." %
                    (for_type, function.func.__name__, cls))

        cls.register(for_type)