def str_(name):
  """Return the string representation of the given 'name'.
  If it is a bytes object, it will be converted into str.
  If it is a str object, it will simply be resurned."""
  if isinstance(name, bytes) and not isinstance(name, str):
    return name.decode('utf8')
  else:
    return name