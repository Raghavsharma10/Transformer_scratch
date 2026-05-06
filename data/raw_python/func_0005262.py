def get_date_range_by_name(name: str, today: datetime=None, tz=None) -> (datetime, datetime):
    """
    :param name: yesterday, last_month
    :param today: Optional current datetime. Default is now().
    :param tz: Optional timezone. Default is UTC.
    :return: datetime (begin, end)
    """
    if today is None:
        today = datetime.utcnow()
    if name == 'last_month':
        return last_month(today, tz)
    elif name == 'last_week':
        return last_week(today, tz)
    elif name == 'this_month':
        return this_month(today, tz)
    elif name == 'last_year':
        return last_year(today, tz)
    elif name == 'yesterday':
        return yesterday(today, tz)
    elif name == 'today':
        begin = today.replace(hour=0, minute=0, second=0, microsecond=0)
        end = begin + timedelta(hours=24)
        return localize_time_range(begin, end, tz)
    else:
        m = re.match(r'^plus_minus_(\d+)d$', name)
        if m:
            days = int(m.group(1))
            return localize_time_range(today - timedelta(days=days), today + timedelta(days=days), tz)
        m = re.match(r'^prev_(\d+)d$', name)
        if m:
            days = int(m.group(1))
            return localize_time_range(today - timedelta(days=days), today, tz)
        m = re.match(r'^next_(\d+)d$', name)
        if m:
            days = int(m.group(1))
            return localize_time_range(today, today + timedelta(days=days), tz)
    raise ValueError('Invalid date range name: {}'.format(name))