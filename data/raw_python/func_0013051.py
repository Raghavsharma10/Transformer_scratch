def _retrieve_value(self, entity, default=None):
    """Internal helper to retrieve the value for this Property from an entity.

    This returns None if no value is set, or the default argument if
    given.  For a repeated Property this returns a list if a value is
    set, otherwise None.  No additional transformations are applied.
    """
    return entity._values.get(self._name, default)