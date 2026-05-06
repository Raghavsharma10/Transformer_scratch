def regex_lexer(regex_pat):
    """
    generate token names' cache
    """

    if isinstance(regex_pat, str):
        regex_pat = re.compile(regex_pat)

        def f(inp_str, pos):
            m = regex_pat.match(inp_str, pos)
            return m.group() if m else None
    elif hasattr(regex_pat, 'match'):
        def f(inp_str, pos):
            m = regex_pat.match(inp_str, pos)
            return m.group() if m else None
    else:
        regex_pats = tuple(re.compile(e) for e in regex_pat)

        def f(inp_str, pos):
            for each_pat in regex_pats:
                m = each_pat.match(inp_str, pos)
                if m:
                    return m.group()

    return f