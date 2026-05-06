def encode_non_ascii_string(string):
    """
    :param string:
        The string to be encoded
    :type string:
        unicode or str
    :return:
        The encoded string
    :rtype:
        str
    """
    encoded_string = string.encode('utf-8', 'replace')
    if six.PY3:
        encoded_string = encoded_string.decode()

    return encoded_string