def enbase64(byte_str):
    """
    Encode bytes/strings to base64.

    Args:
        - ``byte_str``:  The string or bytes to base64 encode.

    Returns:
        - byte_str encoded as base64.
    """

    # Python 3: base64.b64encode() expects type byte
    if isinstance(byte_str, str) and not PYTHON2:
        byte_str = bytes(byte_str, 'utf-8')
    return base64.b64encode(byte_str)