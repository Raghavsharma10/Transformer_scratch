def walk(value, walker, path=None, seen=None):
  """Walks the _evaluated_ tree of the given GCL tuple.

  The appropriate methods of walker will be invoked for every element in the
  tree.
  """
  seen = seen or set()
  path = path or []

  # Recursion
  if id(value) in seen:
    walker.visitRecursion(path)
    return

  # Error
  if isinstance(value, Exception):
    walker.visitError(path, value)
    return

  # List
  if isinstance(value, list):
    # Not actually a tuple, but okay
    recurse = walker.enterList(value, path)
    if not recurse: return
    next_walker = walker if recurse is True else recurse

    with TempSetAdd(seen, id(value)):
      for i, x in enumerate(value):
        walk(x, next_walker, path=path + ['[%d]' % i], seen=seen)

      walker.leaveList(value, path)
    return

  # Scalar
  if not isinstance(value, framework.TupleLike):
    walker.visitScalar(path, value)
    return

  # Tuple
  recurse = walker.enterTuple(value, path)
  if not recurse: return
  next_walker = walker if recurse is True else recurse

  with TempSetAdd(seen, id(value)):
    keys = sorted(value.keys())
    for key in keys:
      key_path = path + [key]
      elm = get_or_error(value, key)
      walk(elm, next_walker, path=key_path, seen=seen)

    walker.leaveTuple(value, path)