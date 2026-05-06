def can_use_enum(func):
    """
    Decorator to use Enum value on type checks.
    """

    @wraps(func)
    def inner(self, value):
        if isinstance(value, Enum):
            return self.check_value(value.value) or func(self, value.value)

        return func(self, value)

    return inner