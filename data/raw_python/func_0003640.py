def cal_gpa(grades):
    """
    根据成绩数组计算课程平均绩点和 gpa, 算法不一定与学校一致, 结果仅供参考

    :param grades: :meth:`models.StudentSession.get_my_achievements` 返回的成绩数组
    :return: 包含了课程平均绩点和 gpa 的元组
    """
    # 课程总数
    courses_sum = len(grades)
    # 课程绩点和
    points_sum = 0
    # 学分和
    credit_sum = 0
    # 课程学分 x 课程绩点之和
    gpa_points_sum = 0
    for grade in grades:
        point = get_point(grade.get('补考成绩') or grade['成绩'])
        credit = float(grade['学分'])

        points_sum += point
        credit_sum += credit
        gpa_points_sum += credit * point
    ave_point = points_sum / courses_sum
    gpa = gpa_points_sum / credit_sum
    return round(ave_point, 5), round(gpa, 5)