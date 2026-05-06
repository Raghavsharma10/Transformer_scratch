def pattern(name, pattern):
  """Function to put a name on a pyparsing pattern.

  Just for ease of debugging/tracing parse errors.
  """
  pattern.setName(name)
  astracing.maybe_trace(pattern)
  return pattern