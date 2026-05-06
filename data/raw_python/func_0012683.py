def _to_rfc822(date):
    """_to_rfc822(datetime.datetime) -> str
    The datetime `strftime` method is subject to locale-specific
    day and month names, so this function hardcodes the conversion."""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fmt = '{day}, {d:02} {month} {y:04} {h:02}:{m:02}:{s:02} GMT'
    return fmt.format(
        day=days[date.weekday()],
        d=date.day,
        month=months[date.month - 1],
        y=date.year,
        h=date.hour,
        m=date.minute,
        s=date.second,
    )