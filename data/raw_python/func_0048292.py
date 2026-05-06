def time_seg_to_index(time_str, slide_step):
    """
    将时间片字符串转换为时间片索引值
    :param time_str: eg: '11:10:21'
    :param slide_step:
    :return:
    """
    minutes_idx = time_util.time_str_to_minutes(time_str)
    time_seg_idx = minutes_idx // slide_step
    return time_seg_idx