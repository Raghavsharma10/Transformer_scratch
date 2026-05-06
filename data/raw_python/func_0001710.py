def read_rcfile():
    """
    Try to read a rcfile from a list of paths
    """
    files = [
        '{}/.millipederc'.format(os.environ.get('HOME')),
        '/usr/local/etc/millipederc',
        '/etc/millipederc',
    ]
    for filepath in files:
        if os.path.isfile(filepath):
            with open(filepath) as rcfile:
                return parse_rcfile(rcfile)
    return {}