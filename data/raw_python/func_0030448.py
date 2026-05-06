def set_url_part(url, **kwargs):
    """Change one or more parts of a URL"""
    d = parse_url_to_dict(url)

    d.update(kwargs)

    return unparse_url_dict(d)