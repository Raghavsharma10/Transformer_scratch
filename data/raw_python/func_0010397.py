def parse_requests_response(response, **kwargs):
    """Build a ContentDisposition from a requests (PyPI) response.
    """

    return parse_headers(
        response.headers.get('content-disposition'), response.url, **kwargs)