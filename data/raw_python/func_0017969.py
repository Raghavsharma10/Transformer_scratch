def ldSet(self, what, key, value):
    """List/dictionary-aware set."""
    if isListKey(key):
      # Make sure we keep the indexes consistent, insert missing_values
      # as necessary. We do remember the lists, so that we can remove
      # missing values after inserting all values from all selectors.
      self.lists[id(what)] = what
      ix = listKeyIndex(key)
      while len(what) <= ix:
        what.append(missing_value)
      what[ix] = value
    else:
      what[key] = value
    return value