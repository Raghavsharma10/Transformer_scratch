def to_python(value, seen=None):
  """Reify values to their Python equivalents.

  Does recursion detection, failing when that happens.
  """
  seen = seen or set()
  if isinstance(value, framework.TupleLike):
    if value.ident in seen:
      raise RecursionException('to_python: infinite recursion while evaluating %r' % value)
    new_seen = seen.union([value.ident])
    return {k: to_python(value[k], seen=new_seen) for k in value.exportable_keys()}
  if isinstance(value, dict):
    return {k: to_python(value[k], seen=seen) for k in value.keys()}
  if isinstance(value, list):
    return [to_python(x, seen=seen) for x in value]
  return value