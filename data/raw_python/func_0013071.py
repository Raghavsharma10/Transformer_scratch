def _set_attributes(self, kwds):
    """Internal helper to set attributes from keyword arguments.

    Expando overrides this.
    """
    cls = self.__class__
    for name, value in kwds.iteritems():
      prop = getattr(cls, name)  # Raises AttributeError for unknown properties.
      if not isinstance(prop, Property):
        raise TypeError('Cannot set non-property %s' % name)
      prop._set_value(self, value)