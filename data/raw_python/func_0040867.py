def get_url_params(url: str, fragment: bool = False) -> dict:
    """
    Parse URL params
    """
    parsed_url = urlparse(url)
    if fragment:
        url_query = parse_qsl(parsed_url.fragment)
    else:
        url_query = parse_qsl(parsed_url.query)
    return dict(url_query)