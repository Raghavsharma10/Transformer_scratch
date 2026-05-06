def get_auth():
    """Return a tuple for authenticating a user

    If not successful raise ``AgileError``.
    """
    auth = get_auth_from_env()
    if auth[0] and auth[1]:
        return auth

    home = os.path.expanduser("~")
    config = os.path.join(home, '.gitconfig')
    if not os.path.isfile(config):
        raise GithubException('No .gitconfig available')

    parser = configparser.ConfigParser()
    parser.read(config)
    if 'user' in parser:
        user = parser['user']
        if 'username' not in user:
            raise GithubException('Specify username in %s user '
                                  'section' % config)
        if 'token' not in user:
            raise GithubException('Specify token in %s user section'
                                  % config)
        return user['username'], user['token']
    else:
        raise GithubException('No user section in %s' % config)