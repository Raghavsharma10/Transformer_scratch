def _apply_to_values(self, entity, function):
    """Apply a function to the property value/values of a given entity.

    This retrieves the property value, applies the function, and then
    stores the value back.  For a repeated property, the function is
    applied separately to each of the values in the list.  The
    resulting value or list of values is both stored back in the
    entity and returned from this method.
    """
    value = self._retrieve_value(entity, self._default)
    if self._repeated:
      if value is None:
        value = []
        self._store_value(entity, value)
      else:
        value[:] = map(function, value)
    else:
      if value is not None:
        newvalue = function(value)
        if newvalue is not None and newvalue is not value:
          self._store_value(entity, newvalue)
          value = newvalue
    return value