def _set(self, value):
    """Updates all descendants to a specified value."""
    if self.__is_parent_node():
      for child in self.__sub_counters.itervalues():
        child._set(value)
    else:
      self.__counter = value