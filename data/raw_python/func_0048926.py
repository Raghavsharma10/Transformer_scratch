def time_str_to_minutes(time_str):
    """
    通过时间字符串计算得到这是一天中第多少分钟
    :param time_str: eg: '11:10:00'
    :return: int
    """
    time_arr = time_str.split(":")
    hours = int(time_arr[0])
    minutes = int(time_arr[1])
    return hours * 60 + minutes