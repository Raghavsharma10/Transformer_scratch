def convertAndMake(converter, handler):
  """Convert with location."""
  def convertAction(loc, value):
    return handler(loc, converter(value))
  return convertAction