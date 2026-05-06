def get_representations_of_kind(kind, start=None, end=None):
  """Return all representations of properties of kind in the specified range.

  NOTE: This function does not return unindexed properties.

  Args:
    kind: name of kind whose properties you want.
    start: only return properties >= start if start is not None.
    end: only return properties < end if end is not None.

  Returns:
    A dictionary mapping property names to its list of representations.
  """
  q = Property.query(ancestor=Property.key_for_kind(kind))
  if start is not None and start != '':
    q = q.filter(Property.key >= Property.key_for_property(kind, start))
  if end is not None:
    if end == '':
      return {}
    q = q.filter(Property.key < Property.key_for_property(kind, end))

  result = {}
  for property in q:
    result[property.property_name] = property.property_representation

  return result