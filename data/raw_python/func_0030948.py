def get_url_size(url):
    """Get the size of a URL.

    Note: Uses requests, so it does not work for FTP URLs.

    Source: StackOverflow user "Burhan Khalid".
    (http://stackoverflow.com/a/24585314/5651021)

    Parameters
    ----------
    url : str
        The URL.

    Returns
    -------
    int
        The size of the URL in bytes.
    """
    r = requests.head(url, headers={'Accept-Encoding': 'identity'})
    size = int(r.headers['content-length'])
    return size