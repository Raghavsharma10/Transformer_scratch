def get(self, option, default=undefined, cast=undefined):
        """
        Return the value for option or default if defined.
        """
        if option in self.repository:
            value = self.repository.get(option)
        else:
            value = default

        if isinstance(value, Undefined):
            raise UndefinedValueError('%s option not found and default value was not defined.' % option)

        if isinstance(cast, Undefined):
            cast = lambda v: v  # nop
        elif cast is bool:
            cast = self._cast_boolean

        return cast(value)