def str_matcher(mode):
    """
    generate token strings' cache
    """

    def f_raw(inp_str, pos):
        return unique_literal_cache_pool[mode] if inp_str.startswith(mode, pos) else None

    def f_collection(inp_str, pos):
        for each in mode:
            if inp_str.startswith(each, pos):
                return unique_literal_cache_pool[each]
        return None

    if isinstance(mode, str):
        return f_raw

    if len(mode) is 1:
        mode = mode[0]
        return f_raw

    return f_collection