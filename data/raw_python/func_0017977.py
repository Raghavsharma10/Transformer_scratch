def schema_spec_from_tuple(tup):
  """Return the schema spec from a run-time tuple."""
  if hasattr(tup, 'get_schema_spec'):
    # Tuples have a TupleSchema field that contains a model of the schema
    return schema.from_spec({
        'fields': TupleSchemaAccess(tup),
        'required': tup.get_required_fields()})
  return schema.AnySchema()