def authenticate(api_key, api_url, **kwargs):
    """Returns a muddle instance, with API key and url set for requests."""

    muddle = Muddle(**kwargs)

    # Login.
    muddle.authenticate(api_key, api_url)
    return muddle