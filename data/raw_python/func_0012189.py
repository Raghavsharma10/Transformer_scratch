def api_walk(uri, per_page=100, key="login"):
    """
    For a GitHub URI, walk all the pages until there's no more content
    """
    page = 1
    result = []

    while True:
        response = get_json(uri + "?page=%d&per_page=%d" % (page, per_page))
        if len(response) == 0:
            break
        else:
            page += 1
            for r in response:
                if key == USER_LOGIN:
                    result.append(user_login(r))
                else:
                    result.append(r[key])

    return list(set(result))