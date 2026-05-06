def _is_initialized(self, entity):
    """Internal helper to ask if the entity has a value for this Property.

    This returns False if a value is stored but it is None.
    """
    return (not self._required or
            ((self._has_value(entity) or self._default is not None) and
             self._get_value(entity) is not None))