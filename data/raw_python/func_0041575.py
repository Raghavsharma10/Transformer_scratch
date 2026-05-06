def unicode2Date(value, format=None):
    """
    CONVERT UNICODE STRING TO UNIX TIMESTAMP VALUE
    """
    # http://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior
    if value == None:
        return None

    if format != None:
        try:
            if format.endswith("%S.%f") and "." not in value:
                value += ".000"
            return _unix2Date(datetime2unix(datetime.strptime(value, format)))
        except Exception as e:
            from mo_logs import Log

            Log.error("Can not format {{value}} with {{format}}", value=value, format=format, cause=e)

    value = value.strip()
    if value.lower() == "now":
        return _unix2Date(datetime2unix(_utcnow()))
    elif value.lower() == "today":
        return _unix2Date(math.floor(datetime2unix(_utcnow()) / 86400) * 86400)
    elif value.lower() in ["eod", "tomorrow"]:
        return _unix2Date(math.floor(datetime2unix(_utcnow()) / 86400) * 86400 + 86400)

    if any(value.lower().find(n) >= 0 for n in ["now", "today", "eod", "tomorrow"] + list(MILLI_VALUES.keys())):
        return parse_time_expression(value)

    try:  # 2.7 DOES NOT SUPPORT %z
        local_value = parse_date(value)  #eg 2014-07-16 10:57 +0200
        return _unix2Date(datetime2unix((local_value - local_value.utcoffset()).replace(tzinfo=None)))
    except Exception as e:
        e = Except.wrap(e)  # FOR DEBUGGING
        pass

    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f"
    ]
    for f in formats:
        try:
            return _unix2Date(datetime2unix(datetime.strptime(value, f)))
        except Exception:
            pass



    deformats = [
        "%Y-%m",# eg 2014-07-16 10:57 +0200
        "%Y%m%d",
        "%d%m%Y",
        "%d%m%y",
        "%d%b%Y",
        "%d%b%y",
        "%d%B%Y",
        "%d%B%y",
        "%Y%m%d%H%M%S",
        "%Y%m%dT%H%M%S",
        "%d%m%Y%H%M%S",
        "%d%m%y%H%M%S",
        "%d%b%Y%H%M%S",
        "%d%b%y%H%M%S",
        "%d%B%Y%H%M%S",
        "%d%B%y%H%M%S"
    ]
    value = deformat(value)
    for f in deformats:
        try:
            return unicode2Date(value, format=f)
        except Exception:
            pass

    else:
        from mo_logs import Log
        Log.error("Can not interpret {{value}} as a datetime", value=value)