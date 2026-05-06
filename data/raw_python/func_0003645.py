def curriculum2schedule(curriculum, first_day, compress=False, time_table=None):
    """
    将课程表转换为上课时间表, 如果 compress=False 结果是未排序的, 否则为压缩并排序后的上课时间表

    :param curriculum: 课表
    :param first_day: 第一周周一, 如 datetime.datetime(2016, 8, 29)
    :param compress: 压缩连续的课时为一个
    :param time_table: 每天上课的时间表, 形如 ``((start timedelta, end timedelta), ...)`` 的 11 × 2 的矩阵
    :return: [(datetime.datetime, str) ...]
    """
    schedule = []
    time_table = time_table or (
        (timedelta(hours=8), timedelta(hours=8, minutes=50)),
        (timedelta(hours=9), timedelta(hours=9, minutes=50)),
        (timedelta(hours=10, minutes=10), timedelta(hours=11)),
        (timedelta(hours=11, minutes=10), timedelta(hours=12)),
        (timedelta(hours=14), timedelta(hours=14, minutes=50)),
        (timedelta(hours=15), timedelta(hours=15, minutes=50)),
        (timedelta(hours=16), timedelta(hours=16, minutes=50)),
        (timedelta(hours=17), timedelta(hours=17, minutes=50)),
        (timedelta(hours=19), timedelta(hours=19, minutes=50)),
        (timedelta(hours=19, minutes=50), timedelta(hours=20, minutes=40)),
        (timedelta(hours=20, minutes=40), timedelta(hours=21, minutes=30))
    )
    for i, d in enumerate(curriculum):
        for j, cs in enumerate(d):
            for c in cs or []:
                course = '{name}[{place}]'.format(name=c['课程名称'], place=c['课程地点'])
                for week in c['上课周数']:
                    day = first_day + timedelta(weeks=week - 1, days=i)
                    start, end = time_table[j]
                    item = (week, day + start, day + end, course)
                    schedule.append(item)

    schedule.sort()
    if compress:
        new_schedule = [schedule[0]]
        for i in range(1, len(schedule)):
            sch = schedule[i]
            # 同一天的连续课程
            if new_schedule[-1][1].date() == sch[1].date() and new_schedule[-1][3] == sch[3]:
                # 更新结束时间
                old_item = new_schedule.pop()
                # week, start, end, course
                new_item = (old_item[0], old_item[1], sch[2], old_item[3])
            else:
                new_item = sch
            new_schedule.append(new_item)
        return new_schedule

    return schedule