def utcnow_ts():
    """Timestamp version of our utcnow function."""
    if utcnow.override_time is None:
        # NOTE(kgriffs): This is several times faster
        # than going through calendar.timegm(...)
        return int(time.time())

    return calendar.timegm(utcnow().timetuple())