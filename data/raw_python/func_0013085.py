def _check_properties(cls, property_names, require_indexed=True):
    """Internal helper to check the given properties exist and meet specified
    requirements.

    Called from query.py.

    Args:
      property_names: List or tuple of property names -- each being a string,
        possibly containing dots (to address subproperties of structured
        properties).

    Raises:
      InvalidPropertyError if one of the properties is invalid.
      AssertionError if the argument is not a list or tuple of strings.
    """
    assert isinstance(property_names, (list, tuple)), repr(property_names)
    for name in property_names:
      assert isinstance(name, basestring), repr(name)
      if '.' in name:
        name, rest = name.split('.', 1)
      else:
        rest = None
      prop = cls._properties.get(name)
      if prop is None:
        cls._unknown_property(name)
      else:
        prop._check_property(rest, require_indexed=require_indexed)