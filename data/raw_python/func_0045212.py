def tvdb_search_series(
    token, series=None, id_imdb=None, id_zap2it=None, lang="en", cache=True
):
    """ Allows the user to search for a series based on the following parameters

    Online docs: https://api.thetvdb.com/swagger#!/Search/get_search_series
    Note: results a maximum of 100 entries per page, no option for pagination=
    """
    if lang not in TVDB_LANGUAGE_CODES:
        raise MapiProviderException(
            "'lang' must be one of %s" % ",".join(TVDB_LANGUAGE_CODES)
        )
    url = "https://api.thetvdb.com/search/series"
    parameters = {"name": series, "imdbId": id_imdb, "zap2itId": id_zap2it}
    headers = {"Accept-Language": lang, "Authorization": "Bearer %s" % token}
    status, content = _request_json(
        url, parameters, headers=headers, cache=cache
    )
    if status == 401:
        raise MapiProviderException("invalid token")
    elif status == 405:
        raise MapiProviderException(
            "series, id_imdb, id_zap2it parameters are mutually exclusive"
        )
    elif status == 404:
        raise MapiNotFoundException
    elif status != 200 or not content.get("data"):
        raise MapiNetworkException("TVDb down or unavailable?")
    return content