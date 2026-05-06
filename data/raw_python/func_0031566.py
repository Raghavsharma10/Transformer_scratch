def trade_day(dt, cal='US'):
    """
    Latest trading day w.r.t given dt

    Args:
        dt: date of reference
        cal: trading calendar

    Returns:
        pd.Timestamp: last trading day

    Examples:
        >>> trade_day('2018-12-25').strftime('%Y-%m-%d')
        '2018-12-24'
    """
    from xone import calendar

    dt = pd.Timestamp(dt).date()
    return calendar.trading_dates(start=dt - pd.Timedelta('10D'), end=dt, calendar=cal)[-1]