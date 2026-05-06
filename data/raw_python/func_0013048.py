def _do_validate(self, value):
    """Call all validations on the value.

    This calls the most derived _validate() method(s), then the custom
    validator function, and then checks the choices.  It returns the
    value, possibly modified in an idempotent way, or raises an
    exception.

    Note that this does not call all composable _validate() methods.
    It only calls _validate() methods up to but not including the
    first _to_base_type() method, when the MRO is traversed looking
    for _validate() and _to_base_type() methods.  (IOW if a class
    defines both _validate() and _to_base_type(), its _validate()
    is called and then the search is aborted.)

    Note that for a repeated Property this function should be called
    for each item in the list, not for the list as a whole.
    """
    if isinstance(value, _BaseValue):
      return value
    value = self._call_shallow_validation(value)
    if self._validator is not None:
      newvalue = self._validator(self, value)
      if newvalue is not None:
        value = newvalue
    if self._choices is not None:
      if value not in self._choices:
        raise datastore_errors.BadValueError(
            'Value %r for property %s is not an allowed choice' %
            (value, self._name))
    return value