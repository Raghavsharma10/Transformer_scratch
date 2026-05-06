def make_auth_headers():
    """Make the authentication headers needed to use the Appveyor API."""
    if not os.path.exists(".appveyor.token"):
        raise RuntimeError(
            "Please create a file named `.appveyor.token` in the current directory. "
            "You can get the token from https://ci.appveyor.com/api-token"
        )
    with open(".appveyor.token") as f:
        token = f.read().strip()

    headers = {
        'Authorization': 'Bearer {}'.format(token),
    }
    return headers