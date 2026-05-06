def smartparse(cls, toparse, tzinfo=None):
    """Method which uses dateutil.parse and extras to try and parse the string.

    Valid dates are found at:
     http://labix.org/python-dateutil#head-1443e0f14ad5dff07efd465e080d1110920673d8-2

    Other valid formats include:
      "now" or "today"
      "yesterday"
      "tomorrow"
      "5 minutes ago"
      "10 hours ago"
      "10h5m ago"
      "start of yesterday"
      "end of tomorrow"
      "end of 3rd of March"

    Args:
      toparse: The string to parse.
      tzinfo: Timezone for the resultant datetime_tz object should be in.
              (Defaults to your local timezone.)

    Returns:
      New datetime_tz object.

    Raises:
      ValueError: If unable to make sense of the input.
    """
    # Default for empty fields are:
    #  year/month/day == now
    #  hour/minute/second/microsecond == 0
    toparse = toparse.strip()

    if tzinfo is None:
      dt = cls.now()
    else:
      dt = cls.now(tzinfo)

    default = dt.replace(hour=0, minute=0, second=0, microsecond=0)

    # Remove "start of " and "end of " prefix in the string
    if toparse.lower().startswith("end of "):
      toparse = toparse[7:].strip()

      dt += datetime.timedelta(days=1)
      dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
      dt -= datetime.timedelta(microseconds=1)

      default = dt

    elif toparse.lower().startswith("start of "):
      toparse = toparse[9:].strip()

      dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
      default = dt

    # Handle strings with "now", "today", "yesterday", "tomorrow" and "ago".
    # Need to use lowercase
    toparselower = toparse.lower()

    if toparselower in ["now", "today"]:
      pass

    elif toparselower == "yesterday":
      dt -= datetime.timedelta(days=1)

    elif toparselower in ("tomorrow", "tommorrow"):
      # tommorrow is spelled wrong, but code out there might be depending on it
      # working
      dt += datetime.timedelta(days=1)

    elif "ago" in toparselower:
      # Remove the "ago" bit
      toparselower = toparselower[:-3]
      # Replace all "a day and an hour" with "1 day 1 hour"
      toparselower = toparselower.replace("a ", "1 ")
      toparselower = toparselower.replace("an ", "1 ")
      toparselower = toparselower.replace(" and ", " ")

      # Match the following
      # 1 hour ago
      # 1h ago
      # 1 h ago
      # 1 hour ago
      # 2 hours ago
      # Same with minutes, seconds, etc.

      tocheck = ("seconds", "minutes", "hours", "days", "weeks", "months",
                 "years")
      result = {}
      for match in re.finditer("([0-9]+)([^0-9]*)", toparselower):
        amount = int(match.group(1))
        unit = match.group(2).strip()

        for bit in tocheck:
          regex = "^([%s]|((%s)s?))$" % (
              bit[0], bit[:-1])

          bitmatch = re.search(regex, unit)
          if bitmatch:
            result[bit] = amount
            break
        else:
          raise ValueError("Was not able to parse date unit %r!" % unit)

      delta = dateutil.relativedelta.relativedelta(**result)
      dt -= delta

    else:
      # Handle strings with normal datetime format, use original case.
      dt = dateutil.parser.parse(toparse, default=default.asdatetime(),
                                 tzinfos=pytz_abbr.tzinfos)
      if dt is None:
        raise ValueError("Was not able to parse date!")

      if dt.tzinfo is pytz_abbr.unknown:
        dt = dt.replace(tzinfo=None)

      if dt.tzinfo is None:
        if tzinfo is None:
          tzinfo = localtz()
        dt = cls(dt, tzinfo)
      else:
        if isinstance(dt.tzinfo, pytz_abbr.tzabbr):
          abbr = dt.tzinfo
          dt = dt.replace(tzinfo=None)
          dt = cls(dt, abbr.zone, is_dst=abbr.dst)

        dt = cls(dt)

    return dt