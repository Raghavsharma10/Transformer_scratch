def _bits_to_geohash(value):
  """Convert a list of GeoHash bits to a GeoHash."""
  ret = []
  # Get 5 bits at a time
  for i in (value[i:i+5] for i in xrange(0, len(value), 5)):
    # Convert binary to integer
    # Note: reverse here, the slice above doesn't work quite right in reverse.
    total = sum([(bit*2**count) for count,bit in enumerate(i[::-1])])
    ret.append(BASE32MAPR[total])
  # Join the string and return
  return "".join(ret)