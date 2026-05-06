def convert_enum(func):
    """
    Decorator to use Enum value on type casts.
    """

    @wraps(func)
    def inner(self, value):
        try:
            if self.check_value(value.value):
                return value.value
            return func(self, value.value)
        except AttributeError:
            pass

        return func(self, value)

    return inner