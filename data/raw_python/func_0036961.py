def parse_path(path):
    '''Parses an address into directory and port parts.

    The last segment of the address will be checked to see if it matches a port
    specification (i.e. contains a colon followed by text). This will be
    returned separately from the directory parts.

    If a leading / is given, that will be returned as the first directory
    component. All other / characters are removed.

    All leading / characters are condensed into a single leading /.

    Any path components that are . will be removed, as they just point to the
    previous path component. For example, '/localhost/.' will become
    '/localhost'. Any path components that are .. will be removed, along with
    the previous path component. If this renders the path empty, it will be
    replaced with '/'.

    Examples:

    >>> parse_path('localhost:30000/manager/comp0.rtc')
    (['localhost:30000', 'manager', 'comp0.rtc'], None)
    
    >>> parse_path('localhost/manager/comp0.rtc:in')
    (['localhost', 'manager', 'comp0.rtc'], 'in')
    
    >>> parse_path('/localhost/manager/comp0.rtc')
    (['/', 'localhost', 'manager', 'comp0.rtc'], None)
    
    >>> parse_path('/localhost/manager/comp0.rtc:in')
    (['/', 'localhost', 'manager', 'comp0.rtc'], 'in')
    
    >>> parse_path('manager/comp0.rtc')
    (['manager', 'comp0.rtc'], None)
    
    >>> parse_path('comp0.rtc')
    (['comp0.rtc'], None)

    '''
    bits = path.lstrip('/').split('/')
    if not bits:
        raise exceptions.BadPathError(path)

    if bits[-1]:
        bits[-1], port = get_port(bits[-1])
    else:
        port = None
    if path[0] == '/':
        bits = ['/'] + bits
    condensed_bits = []
    for bit in bits:
        if bit == '.':
            continue
        if bit == '..':
            condensed_bits = condensed_bits[:-1]
            continue
        condensed_bits.append(bit)
    if not condensed_bits:
        condensed_bits = ['/']
    return condensed_bits, port