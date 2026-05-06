def _validate_argument(self, arg):
        """Validate a type or matcher argument to the constructor."""
        if arg is None:
            return arg

        if isinstance(arg, type):
            return InstanceOf(arg)
        if not isinstance(arg, BaseMatcher):
            raise TypeError(
                "argument of %s can be a type or a matcher (got %r)" % (
                    self.__class__.__name__, type(arg)))

        return arg