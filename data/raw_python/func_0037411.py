def load(file, check_version=True):
    """Load VOEvent from file object.

    A simple wrapper to read a file before passing the contents to
    :py:func:`.loads`. Use with an open file object, e.g.::

        with open('/path/to/voevent.xml', 'rb') as f:
            v = vp.load(f)

    Args:
        file (io.IOBase): An open file object (binary mode preferred), see also
            http://lxml.de/FAQ.html :
            "Can lxml parse from file objects opened in unicode/text mode?"

        check_version (bool): (Default=True) Checks that the VOEvent is of a
            supported schema version - currently only v2.0 is supported.
    Returns:
        :py:class:`Voevent`: Root-node of the  etree.
    """
    s = file.read()
    return loads(s, check_version)