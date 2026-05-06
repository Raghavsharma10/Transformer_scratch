def authenticated_session(username, password):
    """
    Given username and password, return an authenticated Yahoo `requests`
    session that can be used for further scraping requests.

    Throw an AuthencationError if authentication fails.
    """
    session = requests.Session()
    session.headers.update(headers())

    response = session.get(url())
    login_path = path(response.text)
    login_url = urljoin(response.url, login_path)
    login_post_data = post_data(response.text, username, password)

    response = session.post(login_url, data=login_post_data)
    if response.headers['connection'] == 'close':
        raise Exception('Authencation failed')

    return session