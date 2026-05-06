def _get_value(self, entity):
    """Compute and store a default value if necessary."""
    value = super(_ClassKeyProperty, self)._get_value(entity)
    if not value:
      value = entity._class_key()
      self._store_value(entity, value)
    return value