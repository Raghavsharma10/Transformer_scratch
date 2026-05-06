def tvdb_series_id(token, id_tvdb, lang="en", cache=True):
    """ Returns a series records that contains all information known about a
    particular series id

    Online docs: api.thetvdb.com/swagger#!/Series/get_series_id=
    """
    if lang not in TVDB_LANGUAGE_CODES:
        raise MapiProviderException(
            "'lang' must be one of %s" % ",".join(TVDB_LANGUAGE_CODES)
        )
    try:
        url = "https://api.thetvdb.com/series/%d" % int(id_tvdb)
    except ValueError:
        raise MapiProviderException("id_tvdb must be numeric")
    headers = {"Accept-Language": lang, "Authorization": "Bearer %s" % token}
    status, content = _request_json(url, headers=headers, cache=cache)
    if status == 401:
        raise MapiProviderException("invalid token")
    elif status == 404:
        raise MapiNotFoundException
    elif status != 200 or not content.get("data"):
        raise MapiNetworkException("TVDb down or unavailable?")
    return content