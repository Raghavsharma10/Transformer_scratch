def _apply_list(self, methods):
    """Return a single callable that applies a list of methods to a value.

    If a method returns None, the last value is kept; if it returns
    some other value, that replaces the last value.  Exceptions are
    not caught.
    """
    def call(value):
      for method in methods:
        newvalue = method(self, value)
        if newvalue is not None:
          value = newvalue
      return value
    return call