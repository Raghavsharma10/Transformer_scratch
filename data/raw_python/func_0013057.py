def _call_shallow_validation(self, value):
    """Call the initial set of _validate() methods.

    This is similar to _call_to_base_type() except it only calls
    those _validate() methods that can be called without needing to
    call _to_base_type().

    An example: suppose the class hierarchy is A -> B -> C ->
    Property, and suppose A defines _validate() only, but B and C
    define _validate() and _to_base_type().  The full list of
    methods called by _call_to_base_type() is::

      A._validate()
      B._validate()
      B._to_base_type()
      C._validate()
      C._to_base_type()

    This method will call A._validate() and B._validate() but not the
    others.
    """
    methods = []
    for method in self._find_methods('_validate', '_to_base_type'):
      if method.__name__ != '_validate':
        break
      methods.append(method)
    call = self._apply_list(methods)
    return call(value)