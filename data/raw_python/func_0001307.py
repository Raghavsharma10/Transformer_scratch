def base64encode(_input=None):
    """Return base64 encoded representation of a string."""
    if PY2:  # pragma: no cover
        return base64.b64encode(_input)
    elif PY3:  # pragma: no cover
        if isinstance(_input, bytes):
            return base64.b64encode(_input).decode('UTF-8')
        elif isinstance(_input, str):
            return base64.b64encode(bytearray(_input, encoding='UTF-8')).decode('UTF-8')