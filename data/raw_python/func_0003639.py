def get_point(grade_str):
    """
    根据成绩判断绩点

    :param grade_str: 一个字符串,因为可能是百分制成绩或等级制成绩
    :return: 成绩绩点
    :rtype: float
    """
    try:
        grade = float(grade_str)
        assert 0 <= grade <= 100
        if 95 <= grade <= 100:
            return 4.3
        elif 90 <= grade < 95:
            return 4.0
        elif 85 <= grade < 90:
            return 3.7
        elif 82 <= grade < 85:
            return 3.3
        elif 78 <= grade < 82:
            return 3.0
        elif 75 <= grade < 78:
            return 2.7
        elif 72 <= grade < 75:
            return 2.3
        elif 68 <= grade < 72:
            return 2.0
        elif 66 <= grade < 68:
            return 1.7
        elif 64 <= grade < 66:
            return 1.3
        elif 60 <= grade < 64:
            return 1.0
        else:
            return 0.0
    except ValueError:
        if grade_str == '优':
            return 3.9
        elif grade_str == '良':
            return 3.0
        elif grade_str == '中':
            return 2.0
        elif grade_str == '及格':
            return 1.2
        elif grade_str in ('不及格', '免修', '未考'):
            return 0.0
        else:
            raise ValueError('{:s} 不是有效的成绩'.format(grade_str))