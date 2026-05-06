def get_whole_assignment_expression(line, varname, seq_type):
    """
    Example:

    line = "x = Container(cargs=(a, b, c))"
    varname = cargs
    delimiter pair = "()"

    return "a, b, c"

    :return:
    """

    tokens = list(tk.generate_tokens(io.StringIO(line).readline))

    if issubclass(seq_type, tuple):
        L, R = "()"
    elif issubclass(seq_type, list):
        L, R = "[]"
    else:
        raise TypeError("Invalid sequence type given: {}".format(seq_type))

    errmsg = "Unexpected format to process assignment `{}=...` in line '{}'".format(varname, line)

    # Delimiter_open_level
    DOL = 0

    # 0 -> not searching, 1 -> searching for first occurence of `L`, 2 -> searching for last occurence of `R`
    search_mode = 0

    i_start, i_end = None, None

    for i, t in enumerate(tokens):
        if t.type == tk.NAME and t.string == varname:
            search_mode = 1
            i_start = i
            assert tokens[i + 1].string == "="
            assert tokens[i + 2].string == L
            continue

        if search_mode < 1 or not t.type == tk.OP:
            continue

        if t.string == L:
            DOL += 1
            search_mode = 2

        if t.string == R:
            DOL -= 1

        if search_mode == 2 and DOL == 0:
            i_end = i
            break
    else:  # no break
        raise ValueError(errmsg)

    substr = line[tokens[i_start].start[1]: tokens[i_end].end[1]]

    try:
        assert substr.count(L) == 1
        assert substr.count(R) == 1
        assert substr.count('"') == 0
        assert substr.count("'") == 0
    except AssertionError:
        raise ValueError(errmsg)

    return substr