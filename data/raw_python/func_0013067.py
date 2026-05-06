def _set_value(self, entity, value):
    """Setter for key attribute."""
    if value is not None:
      value = _validate_key(value, entity=entity)
      value = entity._validate_key(value)
    entity._entity_key = value