def random_alphanum(length):
    """
    Return a random string of ASCII letters and digits.

    :param int length: The length of string to return
    :returns: A random string
    :rtype: str
    """
    charset = string.ascii_letters + string.digits
    return random_string(length, charset)