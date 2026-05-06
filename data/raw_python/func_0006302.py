def decode(value):
  """Decode a geohash. Returns a (lon,lat) pair."""
  assert value, "Invalid geohash: %s"%value
  # Get the GeoHash bits
  bits = _geohash_to_bits(value)
  # Unzip the GeoHash bits.
  lon = bits[0::2]
  lat = bits[1::2]
  # Convert to lat/lon
  return (
    _bits_to_float(lon, lower=-180.0, upper=180.0),
    _bits_to_float(lat)
  )