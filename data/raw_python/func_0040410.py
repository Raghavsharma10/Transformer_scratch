def random_hex(length):
    """
    Return a random hex string.

    :param int length: The length of string to return
    :returns: A random string
    :rtype: str
    """
    charset = ''.join(set(string.hexdigits.lower()))
    return random_string(length, charset)