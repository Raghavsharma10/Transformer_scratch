def _args_to_val(func, args):
  """Helper for GQL parsing to extract values from GQL expressions.

  This can extract the value from a GQL literal, return a Parameter
  for a GQL bound parameter (:1 or :foo), and interprets casts like
  KEY(...) and plain lists of values like (1, 2, 3).

  Args:
    func: A string indicating what kind of thing this is.
    args: One or more GQL values, each integer, string, or GQL literal.
  """
  from .google_imports import gql  # Late import, to avoid name conflict.
  vals = []
  for arg in args:
    if isinstance(arg, (int, long, basestring)):
      val = Parameter(arg)
    elif isinstance(arg, gql.Literal):
      val = arg.Get()
    else:
      raise TypeError('Unexpected arg (%r)' % arg)
    vals.append(val)
  if func == 'nop':
    if len(vals) != 1:
      raise TypeError('"nop" requires exactly one value')
    return vals[0]  # May be a Parameter
  pfunc = ParameterizedFunction(func, vals)
  if pfunc.is_parameterized():
    return pfunc
  else:
    return pfunc.resolve({}, {})