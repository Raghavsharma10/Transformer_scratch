def is_active(cache, token):
    """
    Accepts the cache and ID token and checks to see if the profile is
    currently logged in. If so, return the token, otherwise throw a
    NotAuthenticatedException.
    :param cache:
    :param token:
    :return:
    """
    profile = cache.get(token)
    if not profile:
        raise exceptions.NotAuthenticatedException(
            'The token is good, but you are not logged in. Please '
            'try logging in again.')
    return profile