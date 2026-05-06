def str_lexer(mode):
    """
    generate token strings' cache
    """
    cast_to_const = ConstStrPool.cast_to_const

    def f_raw(inp_str, pos):
        return cast_to_const(mode) if inp_str.startswith(mode, pos) else None

    def f_collection(inp_str, pos):
        for each in mode:
            if inp_str.startswith(each, pos):
                return cast_to_const(each)
        return None

    if isinstance(mode, str):
        return f_raw

    if len(mode) is 1:
        mode = mode[0]
        return f_raw

    return f_collection