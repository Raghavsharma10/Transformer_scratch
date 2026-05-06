def gen_slide_seg_list(mm_begin, mm_end, seg_duration, slide_step):
    """
    生成时间片开始时刻列表,时间片以slide_step步长进行滑动
    :param mm_begin:
    :param mm_end:
    :param seg_duration:
    :param slide_step:
    :return:
    """
    seg_begin_list = [i for i in
                      range(mm_begin, mm_end - seg_duration + 1, slide_step)]
    seg_list = list(map(time_util.minutes_to_time_str, seg_begin_list))
    return seg_list