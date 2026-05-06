def separate_string(string):
    """
    >>> separate_string("test <2>")
    (['test ', ''], ['2'])
    """
    string_list = regex.split(r'<(?![!=])', regex.sub(r'>', '<', string))
    return string_list[::2], string_list[1::2]