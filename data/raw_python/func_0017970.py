def ldGet(self, what, key):
    """List-aware get."""
    if isListKey(key):
      return what[listKeyIndex(key)]
    else:
      return what[key]