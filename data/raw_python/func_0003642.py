def term_str2code(term_str):
    """
    将学期字符串转换为对应的学期代码串

    :param term_str: 形如 "2012-2013学年第二学期" 的学期字符串
    :return: 形如 "022" 的学期代码
    """
    result = ENV['TERM_PATTERN'].match(term_str).groups()
    year = int(result[0])
    return cal_term_code(year, result[1] == '一')