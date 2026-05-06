def from_spec(spec):
  """Return a schema object from a spec.

  A spec is either a string for a scalar type, or a list of 0 or 1 specs,
  or a dictionary with two elements: {'fields': { ... }, required: [...]}.
  """
  if spec == '':
    return any_schema

  if framework.is_str(spec):
    # Scalar type
    if spec not in SCALAR_TYPES:
      raise exceptions.SchemaError('Not a valid schema type: %r' % spec)
    return ScalarSchema(spec)

  if framework.is_list(spec):
    return ListSchema(spec[0] if len(spec) else any_schema)

  if framework.is_tuple(spec):
    return TupleSchema(spec.get('fields', {}), spec.get('required', []))

  raise exceptions.SchemaError('Not valid schema spec; %r' % spec)