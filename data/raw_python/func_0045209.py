def tvdb_login(api_key):
    """ Logs into TVDb using the provided api key

    Note: You can register for a free TVDb key at thetvdb.com/?tab=apiregister
    Online docs: api.thetvdb.com/swagger#!/Authentication/post_login=
    """
    url = "https://api.thetvdb.com/login"
    body = {"apikey": api_key}
    status, content = _request_json(url, body=body, cache=False)
    if status == 401:
        raise MapiProviderException("invalid api key")
    elif status != 200 or not content.get("token"):
        raise MapiNetworkException("TVDb down or unavailable?")
    return content["token"]