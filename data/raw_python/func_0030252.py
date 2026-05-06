def _detect_timezone_etc_localtime():
  """Detect timezone based on /etc/localtime file."""
  matches = []
  if os.path.exists("/etc/localtime"):
    f = open("/etc/localtime", "rb")
    localtime = pytz.tzfile.build_tzinfo("/etc/localtime", f)
    f.close()

    # We want to match against the local database because /etc/localtime will
    # be copied from that. Once we have found a name for /etc/localtime, we can
    # use the name to get the "same" timezone from the inbuilt pytz database.

    tzdatabase = _load_local_tzinfo()
    if tzdatabase:
      tznames = tzdatabase.keys()
      tzvalues = tzdatabase.__getitem__
    else:
      tznames = pytz.all_timezones
      tzvalues = _tzinfome

    # See if we can find a "Human Name" for this..
    for tzname in tznames:
      tz = tzvalues(tzname)

      if dir(tz) != dir(localtime):
        continue

      for attrib in dir(tz):
        # Ignore functions and specials
        if callable(getattr(tz, attrib)) or attrib.startswith("__"):
          continue

        # This will always be different
        if attrib == "zone" or attrib == "_tzinfos":
          continue

        if getattr(tz, attrib) != getattr(localtime, attrib):
          break

      # We get here iff break didn't happen, i.e. no meaningful attributes
      # differ between tz and localtime
      else:
        # Try and get a timezone from pytz which has the same name as the zone
        # which matches in the local database.
        if tzname not in pytz.all_timezones:
          warnings.warn("Skipping %s because not in pytz database." % tzname)
          continue

        matches.append(_tzinfome(tzname))

    matches.sort(key=lambda x: x.zone)

    if len(matches) == 1:
      return matches[0]

    if len(matches) > 1:
      warnings.warn("We detected multiple matches for your /etc/localtime. "
                    "(Matches where %s)" % matches)
      return matches[0]
    else:
      warnings.warn("We detected no matches for your /etc/localtime.")

    # Register /etc/localtime as the timezone loaded.
    pytz._tzinfo_cache["/etc/localtime"] = localtime
    return localtime