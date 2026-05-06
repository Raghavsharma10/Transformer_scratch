def merge_datetime(date, time='', date_format='%d/%m/%Y', time_format='%H:%M'):
    """Create ``datetime`` object from date and time strings."""
    day = datetime.strptime(date, date_format)
    if time:
        time = datetime.strptime(time, time_format)
        time = datetime.time(time)
        day = datetime.date(day)
        day = datetime.combine(day, time)
    return day