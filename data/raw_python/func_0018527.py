def read(url, **kwargs):
    """
    Read the contents of a URL into memory, return
    """
    response = open_url(url, **kwargs)
    try:
        return response.read()
    finally:
        response.close()