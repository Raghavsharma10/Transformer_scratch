def _get_value(self, entity):
    """Internal helper to get the value for this Property from an entity.

    For a repeated Property this initializes the value to an empty
    list if it is not set.
    """
    if entity._projection:
      if self._name not in entity._projection:
        raise UnprojectedPropertyError(
            'Property %s is not in the projection' % (self._name,))
    return self._get_user_value(entity)