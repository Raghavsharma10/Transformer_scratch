def _message_to_entity(msg, modelclass):
  """Recursive helper for _to_base_type() to convert a message to an entity.

  Args:
    msg: A Message instance.
    modelclass: A Model subclass.

  Returns:
    An instance of modelclass.
  """
  ent = modelclass()
  for prop_name, prop in modelclass._properties.iteritems():
    if prop._code_name == 'blob_':  # TODO: Devise a cleaner test.
      continue  # That's taken care of later.
    value = getattr(msg, prop_name)
    if value is not None and isinstance(prop, model.StructuredProperty):
      if prop._repeated:
        value = [_message_to_entity(v, prop._modelclass) for v in value]
      else:
        value = _message_to_entity(value, prop._modelclass)
    setattr(ent, prop_name, value)
  return ent