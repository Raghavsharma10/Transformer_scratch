def api_get(uri, key=None):
    """
    Simple API endpoint get, return only the keys we care about
    """
    response = get_json(uri)

    if response:
        if type(response) == list:
            r = response[0]
        elif type(response) == dict:
            r = response

        if type(r) == dict:
            # Special nested value we care about
            if key == USER_LOGIN:
                return user_login(r)
            if key in r:
                return r[key]