def isUrl(urlString):
    """
    Attempts to return whether a given URL string is valid by checking
    for the presence of the URL scheme and netloc using the urlparse
    module, and then using a regex.

    From http://stackoverflow.com/questions/7160737/
    """
    parsed = urlparse.urlparse(urlString)
    urlparseValid = parsed.netloc != '' and parsed.scheme != ''
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)'
        r'+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    return regex.match(urlString) and urlparseValid