def get_uri_name(url):
    """Gets the file name from the end of the URL. Only useful for PyBEL's testing though since it looks specifically
    if the file is from the weird owncloud resources distributed by Fraunhofer"""
    url_parsed = urlparse(url)

    url_parts = url_parsed.path.split('/')

    log.info('url parts: %s', url_parts)

    return url_parts[-1]