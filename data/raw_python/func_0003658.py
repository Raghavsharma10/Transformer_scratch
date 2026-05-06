def parse_course(course_str):
    """
    解析课程表里的课程

    :param course_str: 形如 `单片机原理及应用[新安学堂434 (9-15周)]/数字图像处理及应用[新安学堂434 (1-7周)]/` 的课程表数据
    """
    # 解析课程单元格
    # 所有情况
    # 机械原理[一教416 (1-14周)]/
    # 程序与算法综合设计[不占用教室 (18周)]/
    # 财务管理[一教323 (11-17单周)]/
    # 财务管理[一教323 (10-16双周)]/
    # 形势与政策(4)[一教220 (2,4,6-7周)]/
    p = re.compile(r'(.+?)\[(.+?)\s+\(([\d,-单双]+?)周\)\]/')
    courses = p.findall(course_str)
    results = []
    for course in courses:
        d = {'课程名称': course[0], '课程地点': course[1]}
        # 解析上课周数
        week_str = course[2]
        l = week_str.split(',')
        weeks = []
        for v in l:
            m = re.match(r'(\d+)$', v) or re.match(r'(\d+)-(\d+)$', v) or re.match(r'(\d+)-(\d+)(单|双)$', v)
            g = m.groups()
            gl = len(g)
            if gl == 1:
                weeks.append(int(g[0]))
            elif gl == 2:
                weeks.extend([i for i in range(int(g[0]), int(g[1]) + 1)])
            else:
                weeks.extend([i for i in range(int(g[0]), int(g[1]) + 1, 2)])
        d['上课周数'] = weeks
        results.append(d)
    return results