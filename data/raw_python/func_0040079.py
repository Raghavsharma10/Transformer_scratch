def dateAndDepthFromOoid(ooid):
  """ Extract the encoded date and expected storage depth from an ooid.
  ooid: The ooid from which to extract the info
  returns (datetime(yyyy,mm,dd),depth) if the ooid is in expected format else (None,None)
  """
  year = month = day = None
  try:
    day = int(ooid[-2:])
  except:
    return None,None
  try:
    month = int(ooid[-4:-2])
  except:
    return None,None
  try:
    year = 2000 + int(ooid[-6:-4])
    depth = int(ooid[-7])
    if not depth: depth = oldHardDepth
    return (dt.datetime(year,month,day,tzinfo=UTC),depth)
  except:
    return None,None
  return None,None