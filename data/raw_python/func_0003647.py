def schedule2calendar(schedule, name='课表', using_todo=True):
    """
    将上课时间表转换为 icalendar

    :param schedule: 上课时间表
    :param name: 日历名称
    :param using_todo: 使用 ``icalendar.Todo`` 而不是 ``icalendar.Event`` 作为活动类
    :return: icalendar.Calendar()
    """
    # https://zh.wikipedia.org/wiki/ICalendar
    # http://icalendar.readthedocs.io/en/latest
    # https://tools.ietf.org/html/rfc5545
    cal = icalendar.Calendar()
    cal.add('X-WR-TIMEZONE', 'Asia/Shanghai')
    cal.add('X-WR-CALNAME', name)
    cls = icalendar.Todo if using_todo else icalendar.Event
    for week, start, end, data in schedule:
        # "事件"组件更具通用性, Google 日历不支持"待办"组件
        item = cls(
            SUMMARY='第{:02d}周-{}'.format(week, data),
            DTSTART=icalendar.vDatetime(start),
            DTEND=icalendar.vDatetime(end),
            DESCRIPTION='起始于 {}, 结束于 {}'.format(start.strftime('%H:%M'), end.strftime('%H:%M'))
        )
        now = datetime.now()
        # 这个状态"事件"组件是没有的, 对于待办列表类应用有作用
        # https://tools.ietf.org/html/rfc5545#section-3.2.12
        if using_todo:
            if start < now < end:
                item.add('STATUS', 'IN-PROCESS')
            elif now > end:
                item.add('STATUS', 'COMPLETED')
        cal.add_component(item)
    return cal