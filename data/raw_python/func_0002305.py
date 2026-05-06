def get_oembed_data(url, max_width=None, max_height=None, **params):
    """
    Fetch the OEmbed object, return the response as dictionary.
    """
    if max_width:  params['maxwidth'] = max_width
    if max_height: params['maxheight'] = max_height

    registry = get_oembed_providers()
    return registry.request(url, **params)