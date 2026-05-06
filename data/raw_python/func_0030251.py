def _load_local_tzinfo():
  """Load zoneinfo from local disk."""
  tzdir = os.environ.get("TZDIR", "/usr/share/zoneinfo/posix")

  localtzdata = {}
  for dirpath, _, filenames in os.walk(tzdir):
    for filename in filenames:
      filepath = os.path.join(dirpath, filename)
      name = os.path.relpath(filepath, tzdir)

      f = open(filepath, "rb")
      tzinfo = pytz.tzfile.build_tzinfo(name, f)
      f.close()
      localtzdata[name] = tzinfo

  return localtzdata