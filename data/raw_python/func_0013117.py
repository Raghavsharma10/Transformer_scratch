def _make_model_class(message_type, indexed_fields, **props):
  """Construct a Model subclass corresponding to a Message subclass.

  Args:
    message_type: A Message subclass.
    indexed_fields: A list of dotted and undotted field names.
    **props: Additional properties with which to seed the class.

  Returns:
    A Model subclass whose properties correspond to those fields of
    message_type whose field name is listed in indexed_fields, plus
    the properties specified by the **props arguments.  For dotted
    field names, a StructuredProperty is generated using a Model
    subclass created by a recursive call.

  Raises:
    Whatever _analyze_indexed_fields() raises.
    ValueError if a field name conflicts with a name in **props.
    ValueError if a field name is not valid field of message_type.
    ValueError if an undotted field name designates a MessageField.
  """
  analyzed = _analyze_indexed_fields(indexed_fields)
  for field_name, sub_fields in analyzed.iteritems():
    if field_name in props:
      raise ValueError('field name %s is reserved' % field_name)
    try:
      field = message_type.field_by_name(field_name)
    except KeyError:
      raise ValueError('Message type %s has no field named %s' %
                       (message_type.__name__, field_name))
    if isinstance(field, messages.MessageField):
      if not sub_fields:
        raise ValueError(
            'MessageField %s cannot be indexed, only sub-fields' % field_name)
      sub_model_class = _make_model_class(field.type, sub_fields)
      prop = model.StructuredProperty(sub_model_class, field_name,
                                      repeated=field.repeated)
    else:
      if sub_fields is not None:
        raise ValueError(
            'Unstructured field %s cannot have indexed sub-fields' % field_name)
      if isinstance(field, messages.EnumField):
        prop = EnumProperty(field.type, field_name, repeated=field.repeated)
      elif isinstance(field, messages.BytesField):
        prop = model.BlobProperty(field_name,
                                  repeated=field.repeated, indexed=True)
      else:
        # IntegerField, FloatField, BooleanField, StringField.
        prop = model.GenericProperty(field_name, repeated=field.repeated)
    props[field_name] = prop
  return model.MetaModel('_%s__Model' % message_type.__name__,
                         (model.Model,), props)