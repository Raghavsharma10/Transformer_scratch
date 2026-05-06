def dereference_url(url):
    """
    Makes a HEAD request to find the final destination of a URL after
    following any redirects
    """
    res = open_url(url, method='HEAD')
    res.close()
    return res.url