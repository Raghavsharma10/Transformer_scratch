def ts_to_dt_str(ts, dt_format='%Y-%m-%d %H:%M:%S'):
    """
    时间戳转换为日期字符串
    Args:
        ts: 待转换的时间戳
        dt_format: 目标日期字符串格式

    Returns: 日期字符串

    """
    return datetime.datetime.fromtimestamp(int(ts)).strftime(dt_format)