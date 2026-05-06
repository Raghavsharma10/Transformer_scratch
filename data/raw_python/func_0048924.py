def str_to_datetime(dt_str, fmt='%Y-%m-%d %H:%M:%S'):
    """
    字符串转换为datetime类型数据
    :param dt_str:
    :param fmt:
    :return:
    """
    d_time = datetime.datetime.strptime(dt_str, fmt)
    return d_time