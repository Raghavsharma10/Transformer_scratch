def get_sanitized_endpoint(url):
    """
    Sanitize an endpoint, as removing unneeded parameters
    """
    # sanitize esri
    sanitized_url = url.rstrip()
    esri_string = '/rest/services'
    if esri_string in url:
        match = re.search(esri_string, sanitized_url)
        sanitized_url = url[0:(match.start(0)+len(esri_string))]
    return sanitized_url