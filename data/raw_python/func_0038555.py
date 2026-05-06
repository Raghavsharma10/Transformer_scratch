def handle_url_error(error, endpoint, values):
    """
    Intercept BuildErrors of url_for() using flasks build_error_handler API
    """
    url = overlay_url_for(endpoint, **values)
    if url is None:
        exc_type, exc_value, tb = sys.exc_info()
        if exc_value is error:
            reraise(exc_type, exc_value, tb)
        else:
            raise error
    # url_for will use this result, instead of raising BuildError.
    return url