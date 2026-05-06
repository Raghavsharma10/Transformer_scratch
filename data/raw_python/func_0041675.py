def getTypeStr(_type):
  r"""Gets the string representation of the given type.
  """
  if isinstance(_type, CustomType):
    return str(_type)

  if hasattr(_type, '__name__'):
    return _type.__name__

  return ''