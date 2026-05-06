def ub_to_str(string):
    """
    converts py2 unicode / py3 bytestring into str
    Args:
        string (unicode, byte_string): string to be converted
        
    Returns:
        (str)
    """
    if not isinstance(string, str):
        if six.PY2:
            return str(string)
        else:
            return string.decode()
    return string