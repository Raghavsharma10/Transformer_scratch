def _get_prop_from_modelclass(modelclass, name):
  """Helper for FQL parsing to turn a property name into a property object.

  Args:
    modelclass: The model class specified in the query.
    name: The property name.  This may contain dots which indicate
      sub-properties of structured properties.

  Returns:
    A Property object.

  Raises:
    KeyError if the property doesn't exist and the model clas doesn't
    derive from Expando.
  """
  if name == '__key__':
    return modelclass._key

  parts = name.split('.')
  part, more = parts[0], parts[1:]
  prop = modelclass._properties.get(part)
  if prop is None:
    if issubclass(modelclass, model.Expando):
      prop = model.GenericProperty(part)
    else:
      raise TypeError('Model %s has no property named %r' %
                      (modelclass._get_kind(), part))

  while more:
    part = more.pop(0)
    if not isinstance(prop, model.StructuredProperty):
      raise TypeError('Model %s has no property named %r' %
                      (modelclass._get_kind(), part))
    maybe = getattr(prop, part, None)
    if isinstance(maybe, model.Property) and maybe._name == part:
      prop = maybe
    else:
      maybe = prop._modelclass._properties.get(part)
      if maybe is not None:
        # Must get it this way to get the copy with the long name.
        # (See StructuredProperty.__getattr__() for details.)
        prop = getattr(prop, maybe._code_name)
      else:
        if issubclass(prop._modelclass, model.Expando) and not more:
          prop = model.GenericProperty()
          prop._name = name  # Bypass the restriction on dots.
        else:
          raise KeyError('Model %s has no property named %r' %
                         (prop._modelclass._get_kind(), part))

  return prop