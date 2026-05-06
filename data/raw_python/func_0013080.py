def _get_property_for(self, p, indexed=True, depth=0):
    """Internal helper to get the Property for a protobuf-level property."""
    parts = p.name().split('.')
    if len(parts) <= depth:
      # Apparently there's an unstructured value here.
      # Assume it is a None written for a missing value.
      # (It could also be that a schema change turned an unstructured
      # value into a structured one.  In that case, too, it seems
      # better to return None than to return an unstructured value,
      # since the latter doesn't match the current schema.)
      return None
    next = parts[depth]
    prop = self._properties.get(next)
    if prop is None:
      prop = self._fake_property(p, next, indexed)
    return prop