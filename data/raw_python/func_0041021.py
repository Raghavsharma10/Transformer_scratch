def ensure_utf8(app_name_to_show_on_error: str):
    """
    Python3 uses by default the system set, but it expects it to be ‘utf-8’ to work correctly. This
    can generate problems in reading and writing files and in ``.decode()`` method.
    An example how to 'fix' it:

    nano .bash_profile and add the following:
    export LC_CTYPE=en_US.UTF-8
    export LC_ALL=en_US.UTF-8
    """
    encoding = locale.getpreferredencoding()
    if encoding.lower() != 'utf-8':
        raise OSError('{} works only in UTF-8, but yours is set at {}'.format(app_name_to_show_on_error, encoding))