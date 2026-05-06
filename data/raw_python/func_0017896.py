def validate(obj, schema):
  """Validate an object according to its own AND an externally imposed schema."""
  if not framework.EvaluationContext.current().validate:
    # Short circuit evaluation when disabled
    return obj

  # Validate returned object according to its own schema
  if hasattr(obj, 'tuple_schema'):
    obj.tuple_schema.validate(obj)
  # Validate object according to externally imposed schema
  if schema:
    schema.validate(obj)
  return obj