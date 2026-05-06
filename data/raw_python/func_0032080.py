def getfirstline(file, default):
    """
    Returns the first line of a file.
    """
    with open(file, 'rb') as fh:
        content = fh.readlines()
        if len(content) == 1:
            return content[0].decode('utf-8').strip('\n')

    return default