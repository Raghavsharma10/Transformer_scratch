def _equivalent(self, other):
    """Compare two entities of the same class, excluding keys."""
    if other.__class__ is not self.__class__:  # TODO: What about subclasses?
      raise NotImplementedError('Cannot compare different model classes. '
                                '%s is not %s' % (self.__class__.__name__,
                                                  other.__class_.__name__))
    if set(self._projection) != set(other._projection):
      return False
    # It's all about determining inequality early.
    if len(self._properties) != len(other._properties):
      return False  # Can only happen for Expandos.
    my_prop_names = set(self._properties.iterkeys())
    their_prop_names = set(other._properties.iterkeys())
    if my_prop_names != their_prop_names:
      return False  # Again, only possible for Expandos.
    if self._projection:
      my_prop_names = set(self._projection)
    for name in my_prop_names:
      if '.' in name:
        name, _ = name.split('.', 1)
      my_value = self._properties[name]._get_value(self)
      their_value = other._properties[name]._get_value(other)
      if my_value != their_value:
        return False
    return True