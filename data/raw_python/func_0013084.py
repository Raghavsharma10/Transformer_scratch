def _fix_up_properties(cls):
    """Fix up the properties by calling their _fix_up() method.

    Note: This is called by MetaModel, but may also be called manually
    after dynamically updating a model class.
    """
    # Verify that _get_kind() returns an 8-bit string.
    kind = cls._get_kind()
    if not isinstance(kind, basestring):
      raise KindError('Class %s defines a _get_kind() method that returns '
                      'a non-string (%r)' % (cls.__name__, kind))
    if not isinstance(kind, str):
      try:
        kind = kind.encode('ascii')  # ASCII contents is okay.
      except UnicodeEncodeError:
        raise KindError('Class %s defines a _get_kind() method that returns '
                        'a Unicode string (%r); please encode using utf-8' %
                        (cls.__name__, kind))
    cls._properties = {}  # Map of {name: Property}
    if cls.__module__ == __name__:  # Skip the classes in *this* file.
      return
    for name in set(dir(cls)):
      attr = getattr(cls, name, None)
      if isinstance(attr, ModelAttribute) and not isinstance(attr, ModelKey):
        if name.startswith('_'):
          raise TypeError('ModelAttribute %s cannot begin with an underscore '
                          'character. _ prefixed attributes are reserved for '
                          'temporary Model instance values.' % name)
        attr._fix_up(cls, name)
        if isinstance(attr, Property):
          if (attr._repeated or
              (isinstance(attr, StructuredProperty) and
               attr._modelclass._has_repeated)):
            cls._has_repeated = True
          cls._properties[attr._name] = attr
    cls._update_kind_map()