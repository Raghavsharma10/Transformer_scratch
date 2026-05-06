def escape_queue(s):
    """Escapes the path to a queue, e.g. preserves ~ at the begining.
    """
    if isinstance(s, PosixPath):
        s = unicode_(s)
    elif isinstance(s, bytes):
        s = s.decode('utf-8')
    if s.startswith('~/'):
        return '~/' + shell_escape(s[2:])
    else:
        return shell_escape(s)