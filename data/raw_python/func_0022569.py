def __get_type_args(for_type=None, for_types=None):
        """Parse the arguments and return a tuple of types to implement for.

        Raises:
            ValueError or TypeError as appropriate.
        """
        if for_type:
            if for_types:
                raise ValueError("Cannot pass both for_type and for_types.")
            for_types = (for_type,)
        elif for_types:
            if not isinstance(for_types, tuple):
                raise TypeError("for_types must be passed as a tuple of "
                                "types (classes).")
        else:
            raise ValueError("Must pass either for_type or for_types.")

        return for_types