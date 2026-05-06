def format_number(number):
    """
    >>> format_number(1)
    1
    >>> format_number(22)
    22
    >>> format_number(333)
    333
    >>> format_number(4444)
    '4,444'
    >>> format_number(55555)
    '55,555'
    >>> format_number(666666)
    '666,666'
    >>> format_number(7777777)
    '7,777,777'
    """
    char_list = list(str(number))
    length = len(char_list)
    if length <= 3:
        return number

    result = ''
    if length % 3 != 0:
        while len(char_list) % 3 != 0:
            c = char_list[0]
            result += c
            char_list.remove(c)
        result += ','

    i = 0
    while len(char_list) > 0:
        c = char_list[0]
        result += c
        char_list.remove(c)
        i += 1
        if i % 3 == 0:
            result += ','

    return result[0:-1] if result[-1] == ',' else result