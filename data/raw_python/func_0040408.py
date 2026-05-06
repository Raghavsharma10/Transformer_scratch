def random_string(length, charset):
    """
    Return a random string of the given length from the
    given character set.

    :param int length: The length of string to return
    :param str charset: A string of characters to choose from
    :returns: A random string
    :rtype: str
    """
    n = len(charset)
    return ''.join(charset[random.randrange(n)] for _ in range(length))