def _validate(self, msg):
    """Validate an Enum value.

    Raises:
      TypeError if the value is not an instance of self._message_type.
    """
    if not isinstance(msg, self._message_type):
      raise TypeError('Expected a %s instance for %s property',
                      self._message_type.__name__,
                      self._code_name or self._name)