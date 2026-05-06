def render(value):
    """
    This function finishes the url pattern creation by adding starting
    character ^ end possibly by adding end character at the end

    :param value: naive URL value
    :return: raw string
    """
    # Empty urls
    if not value:  # use case: wild card imports
        return r'^$'

    if value[0] != beginning:
        value = beginning + value

    if value[-1] != end:
        value += end

    return value