def lie_in_seg(dt, time_str, seg_duration):
    """
    判断datetime是否在time_str为起点的时间片内
    :param dt:
    :param time_str: eg: '11:10:21'
    :param seg_duration:
    :return:
    """
    minutes_of_day = time_util.get_minutes_of_day(dt)
    range_begin = time_util.time_str_to_minutes(time_str)
    if range_begin <= minutes_of_day < range_begin + seg_duration:
        return True
    else:
        return False