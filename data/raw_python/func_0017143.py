def login(username=None, password=None, token=None, url=None,
          two_factor_callback=None):
    """Construct and return an authenticated GitHub session.

    This will return a GitHubEnterprise session if a url is provided.

    :param str username: login name
    :param str password: password for the login
    :param str token: OAuth token
    :param str url: (optional), URL of a GitHub Enterprise instance
    :param func two_factor_callback: (optional), function you implement to
        provide the Two Factor Authentication code to GitHub when necessary
    :returns: :class:`GitHub <github3.github.GitHub>`

    """
    g = None

    if (username and password) or token:
        g = GitHubEnterprise(url) if url is not None else GitHub()
        g.login(username, password, token, two_factor_callback)

    return g