def char_matcher(mode):
    """
    a faster way for characters to generate token strings cache
    """

    def f_raw(inp_str, pos):
        return mode if inp_str[pos] is mode else None

    def f_collection(inp_str, pos):
        ch = inp_str[pos]
        for each in mode:
            if ch is each:
                return ch
        return None

    if isinstance(mode, str):
        return f_raw

    if len(mode) is 1:
        mode = mode[0]
        return f_raw

    return f_collection