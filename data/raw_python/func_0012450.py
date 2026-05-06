def get_whitespace(txt):
    """
    Returns a list containing the whitespace to the left and
    right of a string as its two elements
    """

    # if the entire parameter is whitespace
    rall = re.search(r'^([\s])+$', txt)
    if rall:
        tmp = txt.split('\n', 1)
        if len(tmp) == 2:
            return (tmp[0], '\n' + tmp[1])  # left, right
        else:
            return ('', tmp[0])  # left, right
    left = ''
    # find whitespace to the left of the parameter
    rlm = re.search(r'^([\s])+', txt)
    if rlm:
        left = rlm.group(0)
    right = ''
    # find whitespace to the right of the parameter
    rrm = re.search(r'([\s])+$', txt)
    if rrm:
        right = rrm.group(0)
    return (left, right)