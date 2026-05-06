def strip_bewit(url):
    """
    Strips the bewit parameter out of a url.

    Returns (encoded_bewit, stripped_url)

    Raises InvalidBewit if no bewit found.

    :param url:
        The url containing a bewit parameter
    :type url: str
    """
    m = re.search('[?&]bewit=([^&]+)', url)
    if not m:
        raise InvalidBewit('no bewit data found')
    bewit = m.group(1)
    stripped_url = url[:m.start()] + url[m.end():]
    return bewit, stripped_url