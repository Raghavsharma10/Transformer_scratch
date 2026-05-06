def _get_base_value_unwrapped_as_list(self, entity):
    """Like _get_base_value(), but always returns a list.

    Returns:
      A new list of unwrapped base values.  For an unrepeated
      property, if the value is missing or None, returns [None]; for a
      repeated property, if the original value is missing or None or
      empty, returns [].
    """
    wrapped = self._get_base_value(entity)
    if self._repeated:
      if wrapped is None:
        return []
      assert isinstance(wrapped, list)
      return [w.b_val for w in wrapped]
    else:
      if wrapped is None:
        return [None]
      assert isinstance(wrapped, _BaseValue)
      return [wrapped.b_val]