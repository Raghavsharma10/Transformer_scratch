def filter_url(url, **kwargs):
    """filter a URL by returning a URL with only the parts specified in the keywords"""

    d = parse_url_to_dict(url)

    d.update(kwargs)

    return unparse_url_dict({k: v for k, v in list(d.items()) if v})