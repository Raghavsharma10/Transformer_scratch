def get_slide_seg_list_belonged(dt_str, seg_duration, slide_step=1,
                                fmt='%Y-%m-%d %H:%M:%S'):
    """
    获取该时刻所属的所有时间片列表
    :param dt_str: datetime string, eg: 2016-10-31 12:22:11
    :param seg_duration: 时间片长度, unit: minute
    :param slide_step: 滑动步长
    :param fmt: datetime string format
    :return: 时间片列表
    """
    dt = time_util.str_to_datetime(dt_str, fmt)
    day_slide_seg_list = gen_slide_seg_list(
            const.FIRST_MINUTE_OF_DAY, const.MINUTES_IN_A_DAY, seg_duration,
            slide_step)
    return filter(lambda x: lie_in_seg(dt, x, seg_duration), day_slide_seg_list)