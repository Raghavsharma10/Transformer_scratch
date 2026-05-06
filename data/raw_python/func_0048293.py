def index_to_time_seg(time_seg_idx, slide_step):
    """
    将时间片索引值转换为时间片字符串
    :param time_seg_idx:
    :param slide_step:
    :return:
    """
    assert (time_seg_idx * slide_step < const.MINUTES_IN_A_DAY)
    return time_util.minutes_to_time_str(time_seg_idx * slide_step)