def validate_urls(urls, allowed_response_codes=None):
    """Validates that a list of urls can be opened and each responds with an allowed response code

    urls -- the list of urls to ping
    allowed_response_codes -- a list of response codes that the validator will ignore
    """

    for url in urls:
        validate_url(url, allowed_response_codes=allowed_response_codes)
    return True