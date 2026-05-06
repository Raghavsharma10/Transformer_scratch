def has_provider_support(provider, media_type):
    """ Verifies if API provider has support for requested media type
    """
    if provider.lower() not in API_ALL:
        return False
    provider_const = "API_" + media_type.upper()
    return provider in globals().get(provider_const, {})