def _to_dict(self, include=None, exclude=None):
    """Return a dict containing the entity's property values.

    Args:
      include: Optional set of property names to include, default all.
      exclude: Optional set of property names to skip, default none.
        A name contained in both include and exclude is excluded.
    """
    if (include is not None and
        not isinstance(include, (list, tuple, set, frozenset))):
      raise TypeError('include should be a list, tuple or set')
    if (exclude is not None and
        not isinstance(exclude, (list, tuple, set, frozenset))):
      raise TypeError('exclude should be a list, tuple or set')
    values = {}
    for prop in self._properties.itervalues():
      name = prop._code_name
      if include is not None and name not in include:
        continue
      if exclude is not None and name in exclude:
        continue
      try:
        values[name] = prop._get_for_dict(self)
      except UnprojectedPropertyError:
        pass  # Ignore unprojected properties rather than failing.
    return values