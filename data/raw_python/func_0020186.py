def is_standalone(text, start, end):
    """check if the string text[start:end] is standalone by checking forwards
    and backwards for blankspaces
    :text: TODO
    :(start, end): TODO
    :returns: the start of next index after text[start:end]

    """
    left = False
    start -= 1
    while start >= 0 and text[start] in spaces_not_newline:
        start -= 1

    if start < 0 or text[start] == '\n':
        left = True

    right = re_space.match(text, end)
    return (start+1, right.end()) if left and right else None