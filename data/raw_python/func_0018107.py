def has_key(tup, key):
  """has(tuple, string) -> bool

  Return whether a given tuple has a key and the key is bound.
  """
  if isinstance(tup, framework.TupleLike):
    return tup.is_bound(key)
  if isinstance(tup, dict):
    return key in tup
  if isinstance(tup, list):
    if not isinstance(key, int):
      raise ValueError('Key must be integer when checking list index')
    return key < len(tup)
  raise ValueError('Not a tuple-like object: %r' % tup)