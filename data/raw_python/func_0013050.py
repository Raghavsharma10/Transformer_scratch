def _set_value(self, entity, value):
    """Internal helper to set a value in an entity for a Property.

    This performs validation first.  For a repeated Property the value
    should be a list.
    """
    if entity._projection:
      raise ReadonlyPropertyError(
          'You cannot set property values of a projection entity')
    if self._repeated:
      if not isinstance(value, (list, tuple, set, frozenset)):
        raise datastore_errors.BadValueError('Expected list or tuple, got %r' %
                                             (value,))
      value = [self._do_validate(v) for v in value]
    else:
      if value is not None:
        value = self._do_validate(value)
    self._store_value(entity, value)