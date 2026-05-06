def read_file(filename):
    """Read a file into a string"""
    p = path.abspath(path.dirname(__file__))
    filepath = path.join(p, filename)
    try:
        return open(filepath).read()
    except IOError:
        return ''