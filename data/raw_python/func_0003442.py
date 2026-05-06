def raise_ssl_error(code, nested=None):
    """Raise an SSL error with the given error code"""
    err_string = str(code) + ": " + _ssl_errors[code]
    if nested:
        raise SSLError(code, err_string + str(nested))
    raise SSLError(code, err_string)