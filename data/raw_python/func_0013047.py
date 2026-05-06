def _IN(self, value):
    """Comparison operator for the 'in' comparison operator.

    The Python 'in' operator cannot be overloaded in the way we want
    to, so we define a method.  For example::

      Employee.query(Employee.rank.IN([4, 5, 6]))

    Note that the method is called ._IN() but may normally be invoked
    as .IN(); ._IN() is provided for the case you have a
    StructuredProperty with a model that has a Property named IN.
    """
    if not self._indexed:
      raise datastore_errors.BadFilterError(
          'Cannot query for unindexed property %s' % self._name)
    from .query import FilterNode  # Import late to avoid circular imports.
    if not isinstance(value, (list, tuple, set, frozenset)):
      raise datastore_errors.BadArgumentError(
          'Expected list, tuple or set, got %r' % (value,))
    values = []
    for val in value:
      if val is not None:
        val = self._do_validate(val)
        val = self._call_to_base_type(val)
        val = self._datastore_type(val)
      values.append(val)
    return FilterNode(self._name, 'in', values)