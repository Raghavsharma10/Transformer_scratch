def regex_matcher(regex_pat):
    """
    generate token names' cache
    :param regex_pat:
    :return:
    """
    if isinstance(regex_pat, str):
        regex_pat = re.compile(regex_pat)

    def f(inp_str, pos):
        m = regex_pat.match(inp_str, pos)
        return m.group() if m else None

    return f