def authorize(login, password, scopes, note='', note_url='', client_id='',
              client_secret='', two_factor_callback=None):
    """Obtain an authorization token for the GitHub API.

    :param str login: (required)
    :param str password: (required)
    :param list scopes: (required), areas you want this token to apply to,
        i.e., 'gist', 'user'
    :param str note: (optional), note about the authorization
    :param str note_url: (optional), url for the application
    :param str client_id: (optional), 20 character OAuth client key for which
        to create a token
    :param str client_secret: (optional), 40 character OAuth client secret for
        which to create the token
    :param func two_factor_callback: (optional), function to call when a
        Two-Factor Authentication code needs to be provided by the user.
    :returns: :class:`Authorization <Authorization>`

    """
    gh = GitHub()
    gh.login(two_factor_callback=two_factor_callback)
    return gh.authorize(login, password, scopes, note, note_url, client_id,
                        client_secret)