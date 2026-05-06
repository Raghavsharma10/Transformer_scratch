def uuidToOoid(uuid,timestamp=None, depth= None):
  """ Create an ooid from a 32-hex-digit string in regular uuid format.
  uuid: must be uuid in expected format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxx7777777
  timestamp: the year-month-day is encoded in the ooid. If none, use current day
  depth: the expected storage depth is encoded in the ooid. If non, use the defaultDepth
  returns a new opaque id string holding the first 24 digits of the provided uuid and encoded date and depth info
  """
  if not timestamp:
    timestamp = utc_now().date()
  if not depth:
    depth = defaultDepth
  assert depth <= 4 and depth >=1
  return "%s%d%02d%02d%02d" %(uuid[:-7],depth,timestamp.year%100,timestamp.month,timestamp.day)