def __append_url_params(self, url, params):
    """Utility method formatting url request parameters."""
    url_parts = list(urlparse(url))
    query = dict(parse_qsl(url_parts[4]))
    query.update(params)
    url_parts[4] = urlencode(query)

    return urlunparse(url_parts)