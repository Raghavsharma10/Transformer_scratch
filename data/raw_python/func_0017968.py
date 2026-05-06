def deep(self):
    """Return a deep dict of the values selected.

    The leaf values may still be gcl Tuples. Use util.to_python() if you want
    to reify everything to real Python values.
    """
    self.lists = {}
    ret = {}
    for path, value in self.paths_values():
      self.recursiveSet(ret, path, value)
    self.removeMissingValuesFromLists()
    return ret