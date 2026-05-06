def tmdb_movies(api_key, id_tmdb, language="en-US", cache=True):
    """ Lookup a movie item using The Movie Database

    Online docs: developers.themoviedb.org/3/movies
    """
    try:
        url = "https://api.themoviedb.org/3/movie/%d" % int(id_tmdb)
    except ValueError:
        raise MapiProviderException("id_tmdb must be numeric")
    parameters = {"api_key": api_key, "language": language}
    status, content = _request_json(url, parameters, cache=cache)
    if status == 401:
        raise MapiProviderException("invalid API key")
    elif status == 404:
        raise MapiNotFoundException
    elif status != 200 or not any(content.keys()):
        raise MapiNetworkException("TMDb down or unavailable?")
    return content