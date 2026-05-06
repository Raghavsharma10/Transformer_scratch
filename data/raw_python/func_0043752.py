def tolist(obj, flat=True, split=True):
  '''
  Returns `obj` as a list: if it is falsy, returns an empty list; if
  it is a string and `split` is truthy, then it is split into
  substrings using Unix shell semantics; if it is sequence-like, a
  list is returned optionally flattened if `flat` is truthy (see
  :func:`flatten`).
  '''
  # todo: it would be "pretty awesome" if this could auto-detect
  #       comma-separation rather than space-separation
  if not obj:
    return []
  if isseq(obj):
    return flatten(obj) if flat else list(obj)
  if isstr(obj) and split:
    return shlex.split(obj)
  return [obj]