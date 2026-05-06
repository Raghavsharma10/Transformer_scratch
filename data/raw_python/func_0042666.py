def cast(self, value):
        """Cast a value to the type required by the option, if one is set.

        This is used to cast the string values gathered from environment
        variable into their required type.

        Args:
            value: The value to cast.

        Returns:
            The value casted to the expected type for the option.
        """
        # if there is no type set for the option, return the given
        # value unchanged.
        if self.type is None:
            return value

        # cast directly
        if self.type in (str, int, float):
            try:
                return self.type(value)
            except Exception as e:
                raise errors.BisonError(
                    'Failed to cast {} to {}'.format(value, self.type)
                ) from e

        # for bool, can't cast a string, since a string is truthy,
        # so we need to check the value.
        elif self.type == bool:
            return value.lower() == 'true'

        # the option type is currently not supported
        else:
            raise errors.BisonError('Unsupported type for casting: {}'.format(self.type))