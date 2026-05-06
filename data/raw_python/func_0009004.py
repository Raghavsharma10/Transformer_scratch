def _always_unicode(cls, path):
        """
        Ensure the path as retrieved from a Python API, such as :func:`os.listdir`,
        is a proper Unicode string.
        """
        if PY3 or isinstance(path, text_type):
            return path
        return path.decode(sys.getfilesystemencoding(), 'surrogateescape')