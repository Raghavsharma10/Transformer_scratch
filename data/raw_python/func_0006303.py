def encode(lonlat, length=12):
  """Encode a (lon,lat) pair to a GeoHash."""
  assert len(lonlat) == 2, "Invalid lon/lat: %s"%lonlat
  # Half the length for each component.
  length /= 2
  lon = _float_to_bits(lonlat[0], lower=-180.0, upper=180.0, length=length*5)
  lat = _float_to_bits(lonlat[1], lower=-90.0, upper=90.0, length=length*5)
  # Zip the GeoHash bits.
  ret = []
  for a,b in zip(lon,lat):
    ret.append(a)
    ret.append(b)
  return _bits_to_geohash(ret)