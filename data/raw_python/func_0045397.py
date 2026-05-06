def get_session(username, password, pin, cookie_path=COOKIE_PATH):
    """Get a new session."""
    class MoparAuth(AuthBase):  # pylint: disable=too-few-public-methods
        """Authentication wrapper."""

        def __init__(self, username, password, pin, cookie_path):
            """Init."""
            self.username = username
            self.password = password
            self.pin = pin
            self.cookie_path = cookie_path

        def __call__(self, r):
            """No-op."""
            return r

    session = requests.session()
    session.auth = MoparAuth(username, password, pin, cookie_path)
    session.headers.update({'User-Agent': USER_AGENT})
    if os.path.exists(cookie_path):
        _LOGGER.info("cookie found at: %s", cookie_path)
        session.cookies = _load_cookies(cookie_path)
    else:
        _login(session)
    return session