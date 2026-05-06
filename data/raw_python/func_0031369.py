def _regexp(expr, item):
    ''' REGEXP function for Sqlite
    '''
    reg = re.compile(expr)
    return reg.search(item) is not None