def _check_property(self, rest=None, require_indexed=True):
    """Override for Property._check_property().

    Raises:
      InvalidPropertyError if no subproperty is specified or if something
      is wrong with the subproperty.
    """
    if not rest:
      raise InvalidPropertyError(
          'Structured property %s requires a subproperty' % self._name)
    self._modelclass._check_properties([rest], require_indexed=require_indexed)