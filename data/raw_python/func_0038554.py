def replaced_url_for(endpoint, filename=None, **values):
    """
    This function acts as "replacement" for the default url_for() and intercepts if it is a request for bower assets

    If the file is not available in bower, the result is passed to flasks url_for().
    This is useful - but not recommended - for "overlaying" the static directory (see README.rst).
    """
    lookup_result = overlay_url_for(endpoint, filename, **values)

    if lookup_result is not None:
        return lookup_result

    return url_for(endpoint, filename=filename, **values)