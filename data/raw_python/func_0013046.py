def _comparison(self, op, value):
    """Internal helper for comparison operators.

    Args:
      op: The operator ('=', '<' etc.).

    Returns:
      A FilterNode instance representing the requested comparison.
    """
    # NOTE: This is also used by query.gql().
    if not self._indexed:
      raise datastore_errors.BadFilterError(
          'Cannot query for unindexed property %s' % self._name)
    from .query import FilterNode  # Import late to avoid circular imports.
    if value is not None:
      value = self._do_validate(value)
      value = self._call_to_base_type(value)
      value = self._datastore_type(value)
    return FilterNode(self._name, op, value)