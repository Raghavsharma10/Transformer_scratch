def fmt(str, args=None, env=None):
  """fmt(string, [tuple]) -> string

  Interpolate a string, replacing {patterns} with the variables with the same
  name. If given a tuple, use the keys from the tuple to substitute. If not
  given a tuple, uses the current environment as the variable source.
  """
  # Normally, we'd just call str.format(**args), but we only want to evaluate
  # values from the tuple which are actually used in the string interpolation,
  # so we use proxy objects.

  # If no args are given, we're able to take the current environment.
  args = args or env
  proxies = {k: StringInterpolationProxy(args, k) for k in args.keys()}
  return str.format(**proxies)