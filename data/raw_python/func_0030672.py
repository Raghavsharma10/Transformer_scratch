def create_win32tz_map(windows_zones_xml):
  """Creates a map between Windows and Olson timezone names.

  Args:
    windows_zones_xml: The CLDR XML mapping.

  Yields:
    (win32_name, olson_name, comment)
  """
  coming_comment = None
  win32_name = None
  territory = None
  parser = genshi.input.XMLParser(StringIO(windows_zones_xml))
  map_zones = {}
  zone_comments = {}

  for kind, data, _ in parser:
    if kind == genshi.core.START and str(data[0]) == "mapZone":
      attrs = data[1]
      win32_name, territory, olson_name = (
          attrs.get("other"), attrs.get("territory"), attrs.get("type").split(" ")[0])

      map_zones[(win32_name, territory)] = olson_name
    elif kind == genshi.core.END and str(data) == "mapZone" and win32_name:
      if coming_comment:
        zone_comments[(win32_name, territory)] = coming_comment
        coming_comment = None
      win32_name = None
    elif kind == genshi.core.COMMENT:
      coming_comment = data.strip()
    elif kind in (genshi.core.START, genshi.core.END, genshi.core.COMMENT):
      coming_comment = None

  for win32_name, territory in sorted(map_zones):
    yield (win32_name, territory, map_zones[(win32_name, territory)],
           zone_comments.get((win32_name, territory), None))