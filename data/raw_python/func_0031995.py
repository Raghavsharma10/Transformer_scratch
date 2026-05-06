def cache_file(symbol, func, has_date, root, date_type='date'):
    """
    Data file

    Args:
        symbol: symbol
        func: use function to categorize data
        has_date: contains date in data file
        root: root path
        date_type: parameters pass to utils.cur_time, [date, time, time_path, ...]

    Returns:
        str: date file
    """
    cur_mod = sys.modules[func.__module__]
    data_tz = getattr(cur_mod, 'DATA_TZ') if hasattr(cur_mod, 'DATA_TZ') else 'UTC'
    cur_dt = utils.cur_time(typ=date_type, tz=data_tz, trading=False)

    if has_date:
        if hasattr(cur_mod, 'FILE_WITH_DATE'):
            file_fmt = getattr(cur_mod, 'FILE_WITH_DATE')
        else:
            file_fmt = '{root}/{typ}/{symbol}/{cur_dt}.parq'
    else:
        if hasattr(cur_mod, 'FILE_NO_DATE'):
            file_fmt = getattr(cur_mod, 'FILE_NO_DATE')
        else:
            file_fmt = '{root}/{typ}/{symbol}.parq'

    return data_file(
        file_fmt=file_fmt, root=root, cur_dt=cur_dt, typ=func.__name__, symbol=symbol
    )