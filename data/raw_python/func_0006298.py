def _bits_to_float(bits, lower=-90.0, middle=0.0, upper=90.0):
  """Convert GeoHash bits to a float."""
  for i in bits:
    if i:
      lower = middle
    else:
      upper = middle
    middle = (upper + lower) / 2
  return middle