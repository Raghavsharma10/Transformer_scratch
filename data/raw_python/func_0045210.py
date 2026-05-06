def tvdb_refresh_token(token):
    """ Refreshes JWT token

    Online docs: api.thetvdb.com/swagger#!/Authentication/get_refresh_token=
    """
    url = "https://api.thetvdb.com/refresh_token"
    headers = {"Authorization": "Bearer %s" % token}
    status, content = _request_json(url, headers=headers, cache=False)
    if status == 401:
        raise MapiProviderException("invalid token")
    elif status != 200 or not content.get("token"):
        raise MapiNetworkException("TVDb down or unavailable?")
    return content["token"]