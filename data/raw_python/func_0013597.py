def logout(cache):
    """
    Logs out the current session by removing it from the cache. This is
    expected to only occur when a session has
    """
    cache.set(flask.session['auth0_key'], None)
    flask.session.clear()
    return True