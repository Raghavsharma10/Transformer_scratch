def _get_value(self, entity):
    """Override _get_value() to *not* raise UnprojectedPropertyError."""
    value = self._get_user_value(entity)
    if value is None and entity._projection:
      # Invoke super _get_value() to raise the proper exception.
      return super(StructuredProperty, self)._get_value(entity)
    return value