def implement(cls, implementations, for_type=None, for_types=None):
        """Provide protocol implementation for a type.

        Register all implementations of multimethod functions in this
        protocol and add the type into the abstract base class of the
        protocol.

        Arguments:
            implementations: A dict of (function, implementation), where each
                function is multimethod and each implementation is a callable.
            for_type: The concrete type implementations apply to.
            for_types: Same as for_type, but takes a tuple of types.

            You may not supply both for_type and for_types for obvious reasons.

        Raises:
            ValueError for arguments.
            TypeError if not all implementations are provided or if there
                are issues related to polymorphism (e.g. attempting to
                implement a non-multimethod function.
        """
        for type_ in cls.__get_type_args(for_type, for_types):
            cls._implement_for_type(for_type=type_,
                                    implementations=implementations)