def _opt_call_from_base_type(self, value):
    """Call _from_base_type() if necessary.

    If the value is a _BaseValue instance, unwrap it and call all
    _from_base_type() methods.  Otherwise, return the value
    unchanged.
    """
    if isinstance(value, _BaseValue):
      value = self._call_from_base_type(value.b_val)
    return value