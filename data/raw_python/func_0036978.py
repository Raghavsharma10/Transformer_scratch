def load_file(path, encoding, encoding_errors):
    """ Given an existing path, attempt to load it as a unicode string. """
    abs_path = abspath(path)
    if exists(abs_path):
        return read_unicode(abs_path, encoding, encoding_errors)
    raise IOError('File %s does not exist' % (abs_path))