def get_timestamp(time=True, date=True, fmt=None):
    """ Return the current timestamp in machine local time.

    Parameters:
    -----------
    time, date : Boolean
        Flag to include the time or date components, respectively,
        in the output.
    fmt : str, optional
        If passed, will override the time/date choice and use as
        the format string passed to `strftime`.
    """

    time_format = "%H:%M:%S"
    date_format = "%m-%d-%Y"

    if fmt is None:
        if time and date:
            fmt = time_format + " " + date_format
        elif time:
            fmt = time_format
        elif date:
            fmt = date_format
        else:
            raise ValueError("One of `date` or `time` must be True!")

    return datetime.now().strftime(fmt)