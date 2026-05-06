def _casual_timedelta_string(meeting):
    """ Return a casual timedelta string.

    If a meeting starts in 2 hours, 15 minutes, and 32 seconds from now, then
    return just "in 2 hours".

    If a meeting starts in 7 minutes and 40 seconds from now, return just "in 7
    minutes".

    If a meeting starts 56 seconds from now, just return "right now".

    """

    now = datetime.datetime.utcnow()
    mdate = meeting['meeting_date']
    mtime = meeting['meeting_time_start']
    dt_string = "%s %s" % (mdate, mtime)
    meeting_dt = datetime.datetime.strptime(dt_string, "%Y-%m-%d %H:%M:%S")
    relative_td = dateutil.relativedelta.relativedelta(meeting_dt, now)

    denominations = ['years', 'months', 'days', 'hours', 'minutes']
    for denomination in denominations:
        value = getattr(relative_td, denomination)
        if value:
            # If the value is only one, then strip off the plural suffix.
            if value == 1:
                denomination = denomination[:-1]
            return "in %i %s" % (value, denomination)

    return "right now"