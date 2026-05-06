def dt_delta(dt, delta):
    """
    获取dt相隔delta的日期
    :param dt:
    :param delta:
    :return:
    """
    delta_time = datetime.timedelta(days=delta)
    target_date = dt + delta_time
    return target_date