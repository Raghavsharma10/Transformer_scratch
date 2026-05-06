def _reExec(regex, string):
    '''This returns [full match, group1, group2, ...], just like JS.'''
    m = regex.search(string)
    if not m: return None
    return [m.group()] + list(m.groups())