def _check_property(self, rest=None, require_indexed=True):
    """Internal helper to check this property for specific requirements.

    Called by Model._check_properties().

    Args:
      rest: Optional subproperty to check, of the form 'name1.name2...nameN'.

    Raises:
      InvalidPropertyError if this property does not meet the given
      requirements or if a subproperty is specified.  (StructuredProperty
      overrides this method to handle subproperties.)
    """
    if require_indexed and not self._indexed:
      raise InvalidPropertyError('Property is unindexed %s' % self._name)
    if rest:
      raise InvalidPropertyError('Referencing subproperty %s.%s '
                                 'but %s is not a structured property' %
                                 (self._name, rest, self._name))