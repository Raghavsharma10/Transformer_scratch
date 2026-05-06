def parse_tibiadata_datetime(date_dict) -> Optional[datetime.datetime]:
    """Parses time objects from the TibiaData API.

    Time objects are made of a dictionary with three keys:
        date: contains a string representation of the time
        timezone: a string representation of the timezone the date time is based on
        timezone_type: the type of representation used in the timezone key


    Parameters
    ----------
    date_dict: :class:`dict`
        Dictionary representing the time object.

    Returns
    -------
    :class:`datetime.date`, optional
        The represented datetime, in UTC.
    """
    try:
        t = datetime.datetime.strptime(date_dict["date"], "%Y-%m-%d %H:%M:%S.%f")
    except (KeyError, ValueError, TypeError):
        return None

    if date_dict["timezone"] == "CET":
        timezone_offset = 1
    elif date_dict["timezone"] == "CEST":
        timezone_offset = 2
    else:
        return None
    # We subtract the offset to convert the time to UTC
    t = t - datetime.timedelta(hours=timezone_offset)
    return t.replace(tzinfo=datetime.timezone.utc)