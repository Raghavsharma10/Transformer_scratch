def ldContains(self, what, key):
    """List/dictinary/missing-aware contains.

    If the value is a "missing_value", we'll treat it as non-existent
    so it will be overwritten by an empty list/dict when necessary to
    assign child keys.
    """
    if isListKey(key):
      i = listKeyIndex(key)
      return i < len(what) and what[i] != missing_value
    else:
      return key in what and what[key] != missing_value