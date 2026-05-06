def syspath(path, prefix=True):
    """Convert a path for use by the operating system. In particular,
    paths on Windows must receive a magic prefix and must be converted
    to Unicode before they are sent to the OS. To disable the magic
    prefix on Windows, set `prefix` to False---but only do this if you
    *really* know what you're doing.
    """
    # Don't do anything if we're not on windows
    if os.path.__name__ != 'ntpath':
        return path

    if not isinstance(path, six.text_type):
        # Beets currently represents Windows paths internally with UTF-8
        # arbitrarily. But earlier versions used MBCS because it is
        # reported as the FS encoding by Windows. Try both.
        try:
            path = path.decode('utf8')
        except UnicodeError:
            # The encoding should always be MBCS, Windows' broken
            # Unicode representation.
            encoding = sys.getfilesystemencoding() or sys.getdefaultencoding()
            path = path.decode(encoding, 'replace')

    # Add the magic prefix if it isn't already there.
    # http://msdn.microsoft.com/en-us/library/windows/desktop/aa365247.aspx
    if prefix and not path.startswith(WINDOWS_MAGIC_PREFIX):
        if path.startswith(u'\\\\'):
            # UNC path. Final path should look like \\?\UNC\...
            path = u'UNC' + path[1:]
        path = WINDOWS_MAGIC_PREFIX + path

    return path