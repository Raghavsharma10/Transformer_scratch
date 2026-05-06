def list_parse(name_list):
    """Parse a comma-separated list of values, or a filename (starting with @)
    containing a list value on each line.
    """

    if name_list and name_list[0] == '@':
        value = name_list[1:]
        if not os.path.exists(value):
            log.warning('The file %s does not exist' % value)
            return
        try:
            return [v.strip() for v in open(value, 'r').readlines()]
        except IOError as e:
            log.warning('reading %s failed: %s; ignoring this file' %
                        (value, e))
    else:
        return [v.strip() for v in name_list.split(',')]