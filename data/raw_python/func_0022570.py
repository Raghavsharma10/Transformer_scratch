def implicit_static(cls, for_type=None, for_types=None):
        """Automatically generate implementations for a type.

        Implement the protocol for the 'for_type' type by dispatching each
        member function of the protocol to an instance method of the same name
        declared on the type 'for_type'.

        Arguments:
            for_type: The type to implictly implement the protocol with.

        Raises:
            TypeError if not all implementations are provided by 'for_type'.
        """
        for type_ in cls.__get_type_args(for_type, for_types):
            implementations = {}
            for function in cls.required():
                method = getattr(type_, function.__name__, None)
                if not callable(method):
                    raise TypeError(
                        "%s.implicit invokation on type %r is missing instance "
                        "method %r."
                        % (cls.__name__, type_, function.__name__))

                implementations[function] = method

            for function in cls.optional():
                method = getattr(type_, function.__name__, None)

                if callable(method):
                    implementations[function] = method

            return cls.implement(for_type=type_,
                                 implementations=implementations)