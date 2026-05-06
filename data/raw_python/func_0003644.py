def filter_curriculum(curriculum, week, weekday=None):
    """
    筛选出指定星期[和指定星期几]的课程

    :param curriculum: 课程表数据
    :param week: 需要筛选的周数, 是一个代表周数的正整数
    :param weekday: 星期几, 是一个代表星期的整数, 1-7 对应周一到周日
    :return: 如果 weekday 参数没给出, 返回的格式与原课表一致, 但只包括了在指定周数的课程, 否则返回指定周数和星期几的当天课程
    """
    if weekday:
        c = [deepcopy(curriculum[weekday - 1])]
    else:
        c = deepcopy(curriculum)
    for d in c:
        l = len(d)
        for t_idx in range(l):
            t = d[t_idx]
            if t is None:
                continue
            # 一般同一时间课程不会重复，重复时给出警告
            t = list(filter(lambda k: week in k['上课周数'], t)) or None
            if t is not None and len(t) > 1:
                logger.warning('第 %d 周周 %d 第 %d 节课有冲突: %s', week, weekday or c.index(d) + 1, t_idx + 1, t)
            d[t_idx] = t
    return c[0] if weekday else c