def debase64(byte_str):
    """
    Decode base64 encoded bytes/strings.

    Args:
        - ``byte_str``:  The string or bytes to base64 encode. 

    Returns:
        - decoded string as type str for python2 and type byte for python3.
    """
    # Python 3: base64.b64decode() expects type byte
    if isinstance(byte_str, str) and not PYTHON2:
        byte_str = bytes(byte_str, 'utf-8')
    return base64.b64decode(byte_str)