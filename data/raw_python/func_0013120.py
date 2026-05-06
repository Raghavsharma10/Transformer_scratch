def _validate(self, value):
    """Validate an Enum value.

    Raises:
      TypeError if the value is not an instance of self._enum_type.
    """
    if not isinstance(value, self._enum_type):
      raise TypeError('Expected a %s instance, got %r instead' %
                      (self._enum_type.__name__, value))