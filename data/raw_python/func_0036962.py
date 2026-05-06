def format_path(path):
    '''Formats a path as a string, placing / between each component.

    @param path A path in rtctree format, as a tuple with the port name as the
                second component.

    Examples:

    >>> format_path((['localhost:30000', 'manager', 'comp0.rtc'], None))
    'localhost:30000/manager/comp0.rtc'

    >>> format_path((['localhost', 'manager', 'comp0.rtc'], 'in'))
    'localhost/manager/comp0.rtc:in'
    
    >>> format_path((['/', 'localhost', 'manager', 'comp0.rtc'], None))
    '/localhost/manager/comp0.rtc'
    
    >>> format_path((['/', 'localhost', 'manager', 'comp0.rtc'], 'in'))
    '/localhost/manager/comp0.rtc:in'

    >>> format_path((['manager', 'comp0.rtc'], None))
    'manager/comp0.rtc'
    
    >>> format_path((['comp0.rtc'], None))
    'comp0.rtc'

    '''
    if path[1]:
        port = ':' + path[1]
    else:
        port = ''
    if type(path[0]) is str:
        # Don't add slashes if the path is singular
        return path[0] + port
    if path[0][0] == '/':
        starter = '/'
        path = path[0][1:]
    else:
        starter = ''
        path = path[0]
    return starter + '/'.join(path) + port