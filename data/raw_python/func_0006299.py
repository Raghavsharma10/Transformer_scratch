def _float_to_bits(value, lower=-90.0, middle=0.0, upper=90.0, length=15):
  """Convert a float to a list of GeoHash bits."""
  ret = []
  for i in range(length):
    if value >= middle:
      lower = middle
      ret.append(1)
    else:
      upper = middle
      ret.append(0)
    middle = (upper + lower) / 2
  return ret