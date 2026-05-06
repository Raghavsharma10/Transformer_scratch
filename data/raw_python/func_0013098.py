def _ReferenceFromSerialized(serialized):
  """Construct a Reference from a serialized Reference."""
  if not isinstance(serialized, basestring):
    raise TypeError('serialized must be a string; received %r' % serialized)
  elif isinstance(serialized, unicode):
    serialized = serialized.encode('utf8')
  return entity_pb.Reference(serialized)