def read_id_list(filepath):
    """
    Get project member id from a file.

    :param filepath: This field is the path of file to read.
    """
    if not filepath:
        return None
    id_list = []
    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            if not re.match('^[0-9]{8}$', line):
                raise('Each line in whitelist or blacklist is expected '
                      'to contain an eight digit ID, and nothing else.')
            else:
                id_list.append(line)
    return id_list