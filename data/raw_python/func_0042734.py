def parse_tibia_datetime(datetime_str) -> Optional[datetime.datetime]:
    """Parses date and time from the format used in Tibia.com

    Accepted format:

    - ``MMM DD YYYY, HH:mm:ss ZZZ``, e.g. ``Dec 10 2018, 21:53:37 CET``.

    Parameters
    -------------
    datetime_str: :class:`str`
        The date and time as represented in Tibia.com

    Returns
    -----------
    :class:`datetime.datetime`, optional
        The represented datetime, in UTC.
    """
    try:
        datetime_str = datetime_str.replace(",", "").replace("&#160;", " ")
        # Extracting timezone
        tz = datetime_str[-4:].strip()

        # Convert time string to time object
        # Removing timezone cause CEST and CET are not supported
        t = datetime.datetime.strptime(datetime_str[:-4].strip(), "%b %d %Y %H:%M:%S")

        # Getting the offset
        if tz == "CET":
            utc_offset = 1
        elif tz == "CEST":
            utc_offset = 2
        else:
            return None
        # Add/subtract hours to get the real time
        t = t - datetime.timedelta(hours=utc_offset)
        return t.replace(tzinfo=datetime.timezone.utc)
    except (ValueError, AttributeError):
        return None