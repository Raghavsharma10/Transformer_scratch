def base64decode(_input=None):
    """Take a base64 encoded string and return the decoded string."""
    missing_padding = 4 - len(_input) % 4
    if missing_padding:
        _input += '=' * missing_padding
    if PY2:  # pragma: no cover
        return base64.decodestring(_input)
    elif PY3:  # pragma: no cover
        if isinstance(_input, bytes):
            return base64.b64decode(_input).decode('UTF-8')
        elif isinstance(_input, str):
            return base64.b64decode(bytearray(_input, encoding='UTF-8')).decode('UTF-8')