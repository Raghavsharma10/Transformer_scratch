def _escape_path(key):
    '''
    Replace '/', '\\' in key
    '''
    return _tobytes(key).replace(b'$', b'$_').replace(b'/', b'$+').replace(b'\\', b'$$')