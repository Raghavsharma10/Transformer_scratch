def pretty_relpath(path, start):
    '''
    Returns a relative path, but only if it doesn't start with a non-pretty parent directory ".."
    '''
    relpath = os.path.relpath(path, start)
    if relpath.startswith('..'):
        return path
    return relpath