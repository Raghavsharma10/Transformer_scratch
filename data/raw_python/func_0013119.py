def _projected_entity_to_message(ent, message_type):
  """Recursive helper for _from_base_type() to convert an entity to a message.

  Args:
    ent: A Model instance.
    message_type: A Message subclass.

  Returns:
    An instance of message_type.
  """
  msg = message_type()
  analyzed = _analyze_indexed_fields(ent._projection)
  for name, sublist in analyzed.iteritems():
    prop = ent._properties[name]
    val = prop._get_value(ent)
    assert isinstance(prop, model.StructuredProperty) == bool(sublist)
    if sublist:
      field = message_type.field_by_name(name)
      assert isinstance(field, messages.MessageField)
      assert prop._repeated == field.repeated
      if prop._repeated:
        assert isinstance(val, list)
        val = [_projected_entity_to_message(v, field.type) for v in val]
      else:
        assert isinstance(val, prop._modelclass)
        val = _projected_entity_to_message(val, field.type)
    setattr(msg, name, val)
  return msg