def get_kinds(start=None, end=None):
  """Return all kinds in the specified range, for the current namespace.

  Args:
    start: only return kinds >= start if start is not None.
    end: only return kinds < end if end is not None.

  Returns:
    A list of kind names between the (optional) start and end values.
  """
  q = Kind.query()
  if start is not None and start != '':
    q = q.filter(Kind.key >= Kind.key_for_kind(start))
  if end is not None:
    if end == '':
      return []
    q = q.filter(Kind.key < Kind.key_for_kind(end))

  return [x.kind_name for x in q]