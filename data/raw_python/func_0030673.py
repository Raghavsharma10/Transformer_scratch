def update_stored_win32tz_map():
  """Downloads the cldr win32 timezone map and stores it in win32tz_map.py."""
  windows_zones_xml = download_cldr_win32tz_map_xml()
  source_hash = hashlib.md5(windows_zones_xml).hexdigest()

  if hasattr(windows_zones_xml, "decode"):
    windows_zones_xml = windows_zones_xml.decode("utf-8")

  map_zones = create_win32tz_map(windows_zones_xml)
  map_dir = os.path.dirname(os.path.abspath(__file__))
  map_filename = os.path.join(map_dir, "win32tz_map.py")
  if os.path.exists(map_filename):
    reload(win32tz_map)
    current_hash = getattr(win32tz_map, "source_hash", None)
    if current_hash == source_hash:
      return False

  map_file = open(map_filename, "w")

  comment = "Map between Windows and Olson timezones taken from %s" % (
      _CLDR_WINZONES_URL,)
  comment2 = "Generated automatically from datetime_tz.py"
  map_file.write("'''%s\n" % comment)
  map_file.write("%s'''\n" % comment2)

  map_file.write("source_hash = '%s' # md5 sum of xml source data\n" % (
      source_hash))

  map_file.write("win32timezones = {\n")
  for win32_name, territory, olson_name, comment in map_zones:
    if territory == '001':
      map_file.write("  %r: %r, # %s\n" % (
          str(win32_name), str(olson_name), comment or ""))
    else:
      map_file.write("  %r: %r, # %s\n" % (
          (str(win32_name), str(territory)), str(olson_name), comment or ""))
  map_file.write("}\n")

  map_file.close()
  return True