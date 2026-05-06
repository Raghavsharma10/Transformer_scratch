def _cast_float(temp_dt):
    """returns utc timestamp"""
    if type(temp_dt) == str:
        fmt = '%Y-%m-%dT%H:%M:00'
        base_dt = temp_dt[0:19]
        tz_offset = eval(temp_dt[19:22])
        temp_dt = datetime.datetime.strptime(base_dt, fmt) - \
                datetime.timedelta(hours=tz_offset)
    return (temp_dt - datetime.datetime(1970, 1, 1)).total_seconds()