def _geohash_to_bits(value):
  """Convert a GeoHash to a list of GeoHash bits."""
  b = map(BASE32MAP.get, value)
  ret = []
  for i in b:
    out = []
    for z in range(5):
      out.append(i & 0b1)
      i = i >> 1
    ret += out[::-1]
  return ret