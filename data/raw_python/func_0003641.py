def cal_term_code(year, is_first_term=True):
    """
    计算对应的学期代码

    :param year: 学年开始年份,例如 "2012-2013学年第二学期" 就是 2012
    :param is_first_term: 是否为第一学期
    :type is_first_term: bool
    :return: 形如 "022" 的学期代码
    """
    if year <= 2001:
        msg = '出现了超出范围年份: {}'.format(year)
        raise ValueError(msg)
    term_code = (year - 2001) * 2
    if is_first_term:
        term_code -= 1
    return '%03d' % term_code