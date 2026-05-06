def numberize(string):
    '''Turns a string into a number (``int`` or ``float``) if it's only a number (ignoring spaces), otherwise returns the string.
    For example, ``"5 "`` becomes ``5`` and ``"2 ton"`` remains ``"2 ton"``'''
    if not isinstance(string,basestring):
        return string
    just_int = r'^\s*[-+]?\d+\s*$'
    just_float = r'^\s*[-+]?\d+\.(\d+)?\s*$'
    if re.match(just_int,string):
        return int(string)
    if re.match(just_float,string):
        return float(string)
    return string