def make_schema_from(value, env):
  """Make a Schema object from the given spec.

  The input and output types of this function are super unclear, and are held together by ponies,
  wishes, duct tape, and a load of tests. See the comments for horrific entertainment.
  """

  # So this thing may not need to evaluate anything[0]
  if isinstance(value, framework.Thunk):
    value = framework.eval(value, env)

  # We're a bit messy. In general, this has evaluated to a Schema object, but not necessarily:
  # for tuples and lists, we still need to treat the objects as specs.
  if isinstance(value, schema.Schema):
    return value

  if framework.is_tuple(value):
    # If it so happens that the thing is a tuple, we need to pass in the data in a bit of a
    # different way into the schema factory (in a dictionary with {fields, required} keys).
    return schema_spec_from_tuple(value)

  if framework.is_list(value):
    # [0] This list may contain tuples, which oughta be treated as specs, or already-resolved schema
    # objects (as returned by 'int' and 'string' literals). make_schema_from
    # deals with both.
    return schema.from_spec([make_schema_from(x, env) for x in value])

  raise exceptions.EvaluationError('Can\'t make a schema from %r' % value)