def _delete_value(self, entity):
    """Internal helper to delete the value for this Property from an entity.

    Note that if no value exists this is a no-op; deleted values will
    not be serialized but requesting their value will return None (or
    an empty list in the case of a repeated Property).
    """
    if self._name in entity._values:
      del entity._values[self._name]