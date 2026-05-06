def _split_regex(regex):
    """
    Return an array of the URL split at each regex match like (?P<id>[\d]+)
    Call with a regex of '^/foo/(?P<id>[\d]+)/bar/$' and you will receive ['/foo/', '/bar/']
    """
    if regex[0] == '^':
        regex = regex[1:]
    if regex[-1] == '$':
        regex = regex[0:-1]
    results = []
    line = ''
    for c in regex:
        if c == '(':
            results.append(line)
            line = ''
        elif c == ')':
            line = ''
        else:
            line = line + c
    if len(line) > 0:
        results.append(line)
    return results