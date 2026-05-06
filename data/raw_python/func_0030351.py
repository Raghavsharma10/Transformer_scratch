def _detect_timezone_windows():
  """Detect timezone on the windows platform."""
  # pylint: disable=global-statement
  global win32timezone_to_en

  # Try and fetch the key_name for the timezone using
  # Get(Dynamic)TimeZoneInformation
  tzi = DTZI_c()
  kernel32 = ctypes.windll.kernel32
  getter = kernel32.GetTimeZoneInformation
  getter = getattr(kernel32, "GetDynamicTimeZoneInformation", getter)

  # code is for daylight savings: 0 means disabled/not defined, 1 means enabled
  # but inactive, 2 means enabled and active
  _ = getter(ctypes.byref(tzi))

  win32tz_key_name = tzi.key_name
  if not win32tz_key_name:
    if win32timezone is None:
      return None
    # We're on Windows before Vista/Server 2008 - need to look up the
    # standard_name in the registry.
    # This will not work in some multilingual setups if running in a language
    # other than the operating system default
    win32tz_name = tzi.standard_name
    if not win32timezone_to_en:
      win32timezone_to_en = dict(
          win32timezone.TimeZoneInfo._get_indexed_time_zone_keys("Std"))
    win32tz_key_name = win32timezone_to_en.get(win32tz_name, win32tz_name)

  territory = locale.getdefaultlocale()[0].split("_", 1)[1]
  olson_name = win32tz_map.win32timezones.get((win32tz_key_name, territory), win32tz_map.win32timezones.get(win32tz_key_name, None))
  if not olson_name:
    return None
  if not isinstance(olson_name, str):
    olson_name = olson_name.encode("ascii")

  return pytz.timezone(olson_name)