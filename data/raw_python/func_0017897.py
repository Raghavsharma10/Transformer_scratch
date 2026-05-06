def attach(obj, schema):
  """Attach the given schema to the given object."""

  # We have a silly exception for lists, since they have no 'attach_schema'
  # method, and I don't feel like making a subclass for List just to add it.
  # So, we recursively search the list for tuples and attach the schema in
  # there.
  if framework.is_list(obj) and isinstance(schema, ListSchema):
    for x in obj:
      attach(x, schema.element_schema)
    return

  # Otherwise, the object should be able to handle its own schema attachment.
  getattr(obj, 'attach_schema', nop)(schema)