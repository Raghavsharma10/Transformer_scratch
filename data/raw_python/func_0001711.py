def parse_rcfile(rcfile):
    """
    Parses rcfile

    Invalid lines are ignored with a warning
    """
    def parse_bool(value):
        """Parse boolean string"""
        value = value.lower()
        if value in ['yes', 'true']:
            return True
        elif value in ['no', 'false']:
            return False
        else:
            raise ValueError('''Can't parse {}'''.format(value))

    valid_keys = {
        'size': int,
        'comment': str,
        'template': str,
        'reverse': parse_bool,
        'opposite': parse_bool,
        'position': int,
    }
    params = {}

    for linenum, line in enumerate(rcfile):
        line = line.strip()
        if not line or line[0] == '#':
            continue
        pos = line.find(' ')
        key = line[:pos]
        value = line[pos:].strip()
        if key in valid_keys.keys():
            try:
                params[key] = valid_keys[key](value)
            except ValueError:
                print('Ignoring line {} from rcfile'.format(linenum + 1),
                      file=sys.stderr)
    return params