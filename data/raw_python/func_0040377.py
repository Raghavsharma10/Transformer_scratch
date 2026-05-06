def getDescendant(Ancestor, RouteParts):
  r"""Resolves a descendant, of the given Ancestor, as pointed by the RouteParts.
  """
  if not RouteParts:
    return Ancestor

  Resolved = Ancestor.Members.get(RouteParts.pop(0))

  if isinstance(Resolved, Group):
    return getDescendant(Resolved, RouteParts)

  else:
    return Resolved