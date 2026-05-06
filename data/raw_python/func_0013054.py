def _opt_call_to_base_type(self, value):
    """Call _to_base_type() if necessary.

    If the value is a _BaseValue instance, return it unchanged.
    Otherwise, call all _validate() and _to_base_type() methods and
    wrap it in a _BaseValue instance.
    """
    if not isinstance(value, _BaseValue):
      value = _BaseValue(self._call_to_base_type(value))
    return value