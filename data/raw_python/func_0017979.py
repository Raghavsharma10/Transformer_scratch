def bracketedList(l, r, sep, expr, allow_missing_close=False):
  """Parse bracketed list.

  Empty list is possible, as is a trailing separator.
  """
  # We may need to backtrack for lists, because of list comprehension, but not for
  # any of the other lists
  strict = l != '['
  closer = sym(r) if not allow_missing_close else p.Optional(sym(r))
  if strict:
    return sym(l) - listMembers(sep, expr) - closer
  else:
    return sym(l) + listMembers(sep, expr) + closer