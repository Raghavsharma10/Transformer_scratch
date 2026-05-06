def strSlist(string):
    """ Converts angle string to signed list. """
    sign = '-' if string[0] == '-' else '+'
    values = [abs(int(x)) for x in string.split(':')]
    return _fixSlist(list(sign) + values)