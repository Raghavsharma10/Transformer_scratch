def createNewOoid(timestamp=None, depth=None):
  """Create a new Ooid for a given time, to be stored at a given depth
  timestamp: the year-month-day is encoded in the ooid. If none, use current day
  depth: the expected storage depth is encoded in the ooid. If non, use the defaultDepth
  returns a new opaque id string holding 24 random hex digits and encoded date and depth info
  """
  if not timestamp:
    timestamp = utc_now().date()
  if not depth:
    depth = defaultDepth
  assert depth <= 4 and depth >=1
  uuid = str(uu.uuid4())
  return "%s%d%02d%02d%02d" %(uuid[:-7],depth,timestamp.year%100,timestamp.month,timestamp.day)