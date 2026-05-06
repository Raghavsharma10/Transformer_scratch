def get_still_seg_belonged(dt_str, seg_duration, fmt='%Y-%m-%d %H:%M:%S'):
    """
    获取该时刻所属的非滑动时间片
    :param dt_str: datetime string, eg: 2016-10-31 12:22:11
    :param seg_duration: 时间片长度, unit: minute
    :param fmt: datetime string format
    :return:
    """
    dt = time_util.str_to_datetime(dt_str, fmt)
    minutes_of_day = time_util.get_minutes_of_day(dt)
    return time_util.minutes_to_time_str(
            minutes_of_day - minutes_of_day % seg_duration)