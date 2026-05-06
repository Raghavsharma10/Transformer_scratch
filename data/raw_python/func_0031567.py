def cur_time(typ='date', tz=DEFAULT_TZ, trading=True, cal='US'):
    """
    Current time

    Args:
        typ: one of ['date', 'time', 'time_path', 'raw', '']
        tz: timezone
        trading: check if current date is trading day
        cal: trading calendar

    Returns:
        relevant current time or date

    Examples:
        >>> cur_dt = pd.Timestamp('now')
        >>> cur_time(typ='date', trading=False) == cur_dt.strftime('%Y-%m-%d')
        True
        >>> cur_time(typ='time', trading=False) == cur_dt.strftime('%Y-%m-%d %H:%M:%S')
        True
        >>> cur_time(typ='time_path', trading=False) == cur_dt.strftime('%Y-%m-%d/%H-%M-%S')
        True
        >>> isinstance(cur_time(typ='raw', tz='Europe/London'), pd.Timestamp)
        True
        >>> isinstance(cur_time(typ='raw', trading=True), pd.Timestamp)
        True
        >>> cur_time(typ='', trading=False) == cur_dt.date()
        True
    """
    dt = pd.Timestamp('now', tz=tz)

    if typ == 'date':
        if trading: return trade_day(dt=dt, cal=cal).strftime('%Y-%m-%d')
        else: return dt.strftime('%Y-%m-%d')

    if typ == 'time': return dt.strftime('%Y-%m-%d %H:%M:%S')
    if typ == 'time_path': return dt.strftime('%Y-%m-%d/%H-%M-%S')
    if typ == 'raw': return dt

    return trade_day(dt).date() if trading else dt.date()